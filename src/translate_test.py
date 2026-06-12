"""Translate a single selected video with a trained model (debug/demo mode)."""

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
from src.rtransformer.recursive_caption_dataset_test import (
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
            step_sizes = raw_batch[1]
            meta       = raw_batch[2]
            batch = [
                prepare_batch_inputs(step_data, device=translator.device)
                for step_data in raw_batch[0]
            ]
            model_inputs = [
                [e["input_ids"]        for e in batch],
                [e["video_feature"]    for e in batch],
                [e["input_mask"]       for e in batch],
                [e["token_type_ids"]   for e in batch],
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
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path,
        max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len,
        max_n_sen=opt.max_n_sen + 10,
        mode=eval_mode,
        recurrent=opt.recurrent,
        untied=opt.untied or opt.mtrans,
        sel_video=opt.sel_video,
    )
    collate_fn = caption_collate if opt.recurrent else single_sentence_collate
    return DataLoader(
        eval_dataset, collate_fn=collate_fn,
        batch_size=opt.batch_size, shuffle=False, num_workers=2,
    )


def _print_predictions(res_dir, sel_video, decoding_strategy):
    """Print ground-truth and predicted sentences for a single video."""
    pred_path = os.path.join(res_dir, "{}_pred_test.json".format(decoding_strategy))
    if not os.path.exists(pred_path):
        logger.warning("Prediction file not found: %s", pred_path)
        return

    raw_data = load_json(pred_path)
    video_results = raw_data.get("results", {}).get(sel_video, [])
    if not video_results:
        logger.warning("No results found for video %s", sel_video)
        return

    gt_text  = "".join(str(s.get("gt_sentence", ""))  for s in video_results)
    pred_text = "".join(
        str(s.get("sentence", b"").split(b" .")[0] + b". ")
        if isinstance(s.get("sentence"), bytes)
        else str(s.get("sentence", "")).split(" .")[0] + ". "
        for s in video_results
    )
    print("\nGT Sentence:  " + gt_text)
    print("Prediction:   " + pred_text + "\n")


def main():
    parser = argparse.ArgumentParser(description="translate_test.py — single-video inference")

    parser.add_argument("--eval_splits", type=str, nargs="+", default=["val"],
                        choices=["val", "test"])
    parser.add_argument("--res_dir", required=True,
                        help="path to dir containing model.chkpt")
    parser.add_argument("--sel_video", type=str, default="v_bXdq2zI1Ms0",
                        help="video name to run inference on")
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

    checkpoint = torch.load(
        os.path.join(opt.res_dir, "model.chkpt"), weights_only=False
    )

    train_opt = checkpoint["opt"]
    for k in train_opt.__dict__:
        if k not in opt.__dict__:
            setattr(opt, k, getattr(train_opt, k))
    logger.info("Loaded train_opt: %s", train_opt)

    decoding_strategy = (
        "beam{}_lp_{}_la_{}".format(
            opt.beam_size, opt.length_penalty_name, opt.length_penalty_alpha
        )
        if opt.use_beam else "greedy"
    )
    save_json(vars(opt),
              os.path.join(opt.res_dir, "{}_eval_cfg.json".format(decoding_strategy)),
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
        translator       = Translator(opt, checkpoint)

        pred_file = os.path.abspath(
            os.path.join(opt.res_dir, "{}_pred_{}.json".format(decoding_strategy, eval_mode))
        )
        if not os.path.exists(pred_file):
            json_res = run_translate(eval_data_loader, translator, opt=opt)
            save_json(json_res, pred_file, save_pretty=True)
        else:
            logger.info("Using existing prediction file at %s", pred_file)

        _print_predictions(opt.res_dir, opt.sel_video, decoding_strategy)
        logger.info("[Info] Finished %s.", eval_mode)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)-10s: %(message)s")
    main()