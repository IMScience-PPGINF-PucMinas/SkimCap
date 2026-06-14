"""Translate input text with trained model."""

import os
import logging
import random
import subprocess
from collections import defaultdict

import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.translator import Translator
from src.rtransformer.recursive_caption_dataset import (
    RecursiveCaptionDataset as RCDataset,
    caption_collate,
    single_sentence_collate,
    prepare_batch_inputs,
)
from src.utils import load_json, merge_dicts, save_json

logger = logging.getLogger(__name__)


def sort_res(res_dict):
    """Sort each video's predictions by start timestamp."""
    return {
        k: sorted(v, key=lambda x: float(x["timestamp"][0]))
        for k, v in res_dict.items()
    }


def run_translate(eval_data_loader, translator, opt):
    batch_res = {
        "version": "VERSION 1.0",
        "results": defaultdict(list),
        "external_data": {"used": "true", "details": "ay"},
    }

    for raw_batch in tqdm(eval_data_loader, mininterval=2, desc="  - (Translate)"):
        if opt.recurrent:
            step_sizes = raw_batch[1]   # list[int], len == bsz
            meta       = raw_batch[2]   # list[dict], len == bsz
            batch = [
                prepare_batch_inputs(step_data, device=translator.device)
                for step_data in raw_batch[0]
            ]
            # Pass CLIP features alongside the standard visual inputs
            model_inputs = [
                [e["input_ids"]       for e in batch],
                [e["video_feature"]   for e in batch],
                [e["input_mask"]      for e in batch],
                [e["token_type_ids"]  for e in batch],
                [e.get("lang_feature") for e in batch],
                [e.get("lang_mask")    for e in batch],
                [e.get("sent_feat")    for e in batch],
            ]

            dec_seq_list = translator.translate_batch(
                model_inputs, use_beam=opt.use_beam,
                recurrent=True, untied=False, xl=opt.xl,
            )

            for example_idx, (step_size, cur_meta) in enumerate(zip(step_sizes, meta)):
                for step_idx, step_batch in enumerate(dec_seq_list[:step_size]):
                    batch_res["results"][cur_meta["name"]].append({
                        "sentence": eval_data_loader.dataset.convert_ids_to_sentence(
                            step_batch[example_idx].cpu().tolist()
                        ).encode("ascii", "ignore").decode("ascii"),
                        "timestamp":   cur_meta["timestamp"][step_idx],
                        "gt_sentence": cur_meta["gt_sentence"][step_idx],
                    })

        else:  # single sentence
            meta         = raw_batch[2]
            batched_data = prepare_batch_inputs(raw_batch[0], device=translator.device)
            if opt.untied or opt.mtrans:
                model_inputs = [
                    batched_data["video_feature"],
                    batched_data["video_mask"],
                    batched_data["text_ids"],
                    batched_data["text_mask"],
                    batched_data["text_labels"],
                ]
            else:
                model_inputs = [
                    batched_data["input_ids"],
                    batched_data["video_feature"],
                    batched_data["input_mask"],
                    batched_data["token_type_ids"],
                ]

            dec_seq = translator.translate_batch(
                model_inputs, use_beam=opt.use_beam,
                recurrent=False, untied=opt.untied or opt.mtrans,
            )

            for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, meta)):
                batch_res["results"][cur_meta["name"]].append({
                    "sentence": eval_data_loader.dataset.convert_ids_to_sentence(
                        cur_gen_sen.cpu().tolist()
                    ).encode("ascii", "ignore").decode("ascii"),
                    "timestamp":   cur_meta["timestamp"],
                    "gt_sentence": cur_meta["gt_sentence"],
                })

        if opt.debug:
            break

    batch_res["results"] = sort_res(batch_res["results"])
    return batch_res


def get_data_loader(opt, eval_mode="val"):
    eval_dataset = RCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir,
        video_feature_dir=opt.video_feature_dir,
        flow_feature_dir=opt.flow_feature_dir,
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path,
        max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len,
        max_n_sen=opt.max_n_sen + 10,
        mode=eval_mode,
        recurrent=opt.recurrent,
        untied=opt.untied or opt.mtrans,
        lang_feature_dir=getattr(opt, "lang_feature_dir", None),
        sent_feature_dir=getattr(opt, "sent_feature_dir", None),
        vocab_clip_path=getattr(opt, "vocab_clip_path", None),
        use_flow=not getattr(opt, "no_flow", False),
        use_lang=not getattr(opt, "no_lang", False),
        use_sent=not getattr(opt, "no_sent", False),
    )
    collate_fn = caption_collate if opt.recurrent else single_sentence_collate
    return DataLoader(
        eval_dataset, collate_fn=collate_fn,
        batch_size=opt.batch_size, shuffle=False, num_workers=2,
    )


def main():
    parser = argparse.ArgumentParser(description="translate.py")

    parser.add_argument("--eval_splits", type=str, nargs="+", default=["val"],
                        choices=["val", "test"],
                        help="evaluate on val/test set (yc2 only has val)")
    parser.add_argument("--res_dir", required=True,
                        help="path to dir containing model checkpoints")
    parser.add_argument("--checkpoint", type=str, default="model.chkpt",
                        help="checkpoint filename inside res_dir (default: model.chkpt). "
                             "E.g. --checkpoint model_epoch_05.chkpt")
    parser.add_argument("--batch_size", type=int, default=100)

    # beam search
    parser.add_argument("--use_beam", action="store_true")
    parser.add_argument("--beam_size", type=int, default=2)
    parser.add_argument("--n_best", type=int, default=1)
    parser.add_argument("--min_sen_len", type=int, default=5)
    parser.add_argument("--max_sen_len", type=int, default=30)
    parser.add_argument("--block_ngram_repeat", type=int, default=0)
    parser.add_argument("--length_penalty_name", default="none",
                        choices=["none", "wu", "avg"])
    parser.add_argument("--length_penalty_alpha", type=float, default=0.)
    parser.add_argument("--eval_tool_dir", type=str, default="./densevid_eval")

    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--debug", action="store_true")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    chkpt_filename = opt.checkpoint if opt.checkpoint else "model.chkpt"
    chkpt_path = os.path.join(opt.res_dir, chkpt_filename)
    if not os.path.isfile(chkpt_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(chkpt_path))
    logger.info("Loading checkpoint: %s", chkpt_path)

    # Stem used to namespace all output files for this checkpoint,
    # e.g. "model_epoch_05.chkpt" -> "model_epoch_05"
    chkpt_stem = os.path.splitext(chkpt_filename)[0]

    checkpoint = torch.load(chkpt_path, weights_only=False)

    # Merge train-time options (without overwriting inference-time ones)
    train_opt = checkpoint["opt"]
    for k in train_opt.__dict__:
        if k not in opt.__dict__:
            setattr(opt, k, getattr(train_opt, k))

    decoding_strategy = (
        "beam{}_lp_{}_la_{}".format(
            opt.beam_size, opt.length_penalty_name, opt.length_penalty_alpha
        )
        if opt.use_beam else "greedy"
    )
    save_json(vars(opt),
              os.path.join(opt.res_dir, "{}_{}_eval_cfg.json".format(chkpt_stem, decoding_strategy)),
              save_pretty=True)

    if opt.dset_name == "anet":
        reference_files_map = {
            "val":  [os.path.join(opt.data_dir, e) for e in
                     ["anet_entities_val_1_para.json", "anet_entities_val_2_para.json"]],
            "test": [os.path.join(opt.data_dir, e) for e in
                     ["anet_entities_test_1_para.json", "anet_entities_test_2_para.json"]],
        }
    else:  # yc2
        reference_files_map = {
            "val": [os.path.join(opt.data_dir, "yc2_val_anet_format_para.json")]
        }

    for eval_mode in opt.eval_splits:
        logger.info("Start evaluating %s", eval_mode)
        eval_data_loader = get_data_loader(opt, eval_mode=eval_mode)
        eval_references  = reference_files_map[eval_mode]

        translator = Translator(opt, checkpoint)

        pred_file = os.path.abspath(
            os.path.join(opt.res_dir, "{}_{}_pred_{}.json".format(chkpt_stem, decoding_strategy, eval_mode))
        )
        if not os.path.exists(pred_file):
            json_res = run_translate(eval_data_loader, translator, opt=opt)
            save_json(json_res, pred_file, save_pretty=True)
        else:
            logger.info("Using existing prediction file at %s", pred_file)

        lang_file     = pred_file.replace(".json", "_lang.json")
        stat_filepath = pred_file.replace(".json", "_stat.json")
        rep_filepath  = pred_file.replace(".json", "_rep.json")

        subprocess.call(
            ["python", "para-evaluate.py", "-s", pred_file, "-o", lang_file, "-v", "-r"]
            + eval_references,
            cwd=opt.eval_tool_dir,
        )
        subprocess.call(
            ["python", "get_caption_stat.py", "-s", pred_file,
             "-r", eval_references[0], "-o", stat_filepath, "-v"],
            cwd=opt.eval_tool_dir,
        )
        subprocess.call(
            ["python", "evaluateRepetition.py", "-s", pred_file,
             "-r", eval_references[0], "-o", rep_filepath],
            cwd=opt.eval_tool_dir,
        )

        all_metrics = merge_dicts([load_json(p) for p in [lang_file, stat_filepath, rep_filepath]])
        save_json(all_metrics, pred_file.replace(".json", "_all_metrics.json"), save_pretty=True)

        logger.info("pred_file %s  lang_file %s", pred_file, lang_file)
        logger.info("[Info] Finished %s.", eval_mode)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)-10s: %(message)s")
    main()