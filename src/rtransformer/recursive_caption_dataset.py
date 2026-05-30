import copy
import torch
import logging
import math
import nltk
import numpy as np
import os

from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.utils import load_json, flat_list_of_lists

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)


class RecursiveCaptionDataset(Dataset):
    """
    recurrent: if True, return recurrent data

    Feature loading:
        - C3D features:  <c3d_feature_dir>/<video_name>.npy          shape (100, 2048)
        - Flow features: <flow_feature_dir>/<video_name>_bn.npy       shape ( 28, 1024)

    Flow is resampled from 28 → 100 clips via linear interpolation so both
    modalities share the same temporal resolution before concatenation.
    Final video_feature shape per clip: (max_v_len, 3072)  [2048 C3D + 1024 flow]

    Because every C3D file already has exactly 100 clips, no skimming or
    timestamp-based indexing is needed: the full feature array is used as-is,
    padded/trimmed only to max_v_len if necessary.
    """

    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"
    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    IGNORE = -1  # used to calculate loss

    def __init__(self, dset_name, data_dir, video_feature_dir, flow_feature_dir, duration_file, word2idx_path,
                 max_t_len, max_v_len, max_n_sen, mode="train", recurrent=True, untied=False):
        self.dset_name = dset_name
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.data_dir = data_dir
        self.duration_file = duration_file
        self.frame_to_second = self._load_duration()
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len
        self.max_n_sen = max_n_sen

        # ── Feature directories ───────────────────────────────────────────────
        # video_feature_dir  → C3D features:  <dir>/<video_name>.npy
        # flow_feature_dir   → Flow features: <dir>/<video_name>_bn.npy
        #
        # Example args:
        #   --video_feature_dir  /path/to/c3d_anet_feature
        #   --flow_feature_dir   /path/to/rt_anet_feat/trainval
        self.c3d_feature_dir = video_feature_dir
        self.flow_feature_dir = flow_feature_dir

        self.mode = mode
        self.recurrent = recurrent
        self.untied = untied
        assert not (self.recurrent and self.untied), "untied and recurrent cannot be True for both"

        self.data = None
        self.set_data_mode(mode=mode)
        self.missing_video_names = []
        self.fix_missing()

        self.num_sens = None

    # ─────────────────────────────────────────────────────────────────────────
    # Feature loading helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _c3d_path(self, video_name: str) -> str:
        # C3D files keep the full name including the "v_" prefix (e.g. v_C4V6fqELvPY.npy)
        return os.path.join(self.c3d_feature_dir, "v_{}.npy".format(video_name))

    def _flow_path(self, video_name: str) -> str:
        # Flow files have no "v_" prefix (e.g. C4V6fqELvPY_bn.npy)
        return os.path.join(self.flow_feature_dir, "{}_bn.npy".format(video_name))

    @staticmethod
    def _resample_flow(flow: np.ndarray, target_len: int) -> np.ndarray:
        """Linearly resample flow from its original length to *target_len*.

        Args:
            flow:       (src_len, 1024) float array
            target_len: desired number of clips (typically 100)

        Returns:
            (target_len, 1024) float32 array
        """
        src_len = flow.shape[0]
        if src_len == target_len:
            return flow.astype(np.float32)

        x_src = np.linspace(0.0, 1.0, src_len)
        x_tgt = np.linspace(0.0, 1.0, target_len)
        f = interp1d(x_src, flow, axis=0, kind="linear", assume_sorted=True)
        return f(x_tgt).astype(np.float32)

    def _load_video_feature(self, video_name: str) -> np.ndarray:
        """Load and concatenate C3D + flow features.

        C3D  : (100, 2048) — used as-is
        Flow : ( 28, 1024) — resampled to (100, 1024)
        Output: (100, 3072) float32
        """
        c3d = np.load(self._c3d_path(video_name)).astype(np.float32)   # (100, 2048)
        flow = np.load(self._flow_path(video_name))                     # ( 28, 1024)
        flow_resampled = self._resample_flow(flow, target_len=c3d.shape[0])
        return np.concatenate([c3d, flow_resampled], axis=1)            # (100, 3072)

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset setup
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta

    def set_data_mode(self, mode):
        """mode: `train` or `val`"""
        logger.info("Mode {}".format(mode))
        self.mode = mode
        if self.dset_name == "anet":
            if mode == "train":
                data_path = os.path.join(self.data_dir, "train.json")
            elif mode == "val":
                data_path = os.path.join(self.data_dir, "anet_entities_val_1.json")
            elif mode == "test":
                data_path = os.path.join(self.data_dir, "anet_entities_test_1.json")
            else:
                raise ValueError("Expecting mode to be one of [`train`, `val`, `test`], got {}".format(mode))
        elif self.dset_name == "yc2":
            if mode == "train":
                data_path = os.path.join(self.data_dir, "yc2_train_anet_format.json")
            elif mode == "val":
                data_path = os.path.join(self.data_dir, "yc2_val_anet_format.json")
            else:
                raise ValueError("Expecting mode to be one of [`train`, `val`, `test`], got {}".format(mode))
        else:
            raise ValueError
        self._load_data(data_path)

    def fix_missing(self):
        """Filter out videos whose C3D or flow feature file is missing."""
        for e in tqdm(self.data):
            video_name = e["name"][2:] if self.dset_name == "anet" else e["name"]
            for p in [self._c3d_path(video_name), self._flow_path(video_name)]:
                if not os.path.exists(p):
                    self.missing_video_names.append(video_name)
        logger.info("Missing {} features (clips/sentences) from {} videos".format(
            len(self.missing_video_names), len(set(self.missing_video_names))))
        logger.info("Missing {}".format(set(self.missing_video_names)))
        if self.dset_name == "anet":
            self.data = [e for e in self.data if e["name"][2:] not in self.missing_video_names]
        else:
            self.data = [e for e in self.data if e["name"] not in self.missing_video_names]

    def _load_duration(self):
        """https://github.com/salesforce/densecap/blob/master/data/anet_dataset.py#L120
        Since the features are extracted not at the exact 0.5 secs. To get the real time for each feature,
        use `(idx + 1) * frame_to_second[vid_name]`
        """
        frame_to_second = {}
        sampling_sec = 0.5  # hard coded, only support 0.5
        if self.dset_name == "anet":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    frame_to_second[vid_name] = float(vid_dur) * int(
                        float(vid_frame) * 1. / int(float(vid_dur)) * sampling_sec) * 1. / float(vid_frame)
                frame_to_second["_0CqozZun3U"] = sampling_sec  # a missing video in anet
        elif self.dset_name == "yc2":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    frame_to_second[vid_name] = float(vid_dur) * math.ceil(
                        float(vid_frame) * 1. / float(vid_dur) * sampling_sec) * 1. / float(vid_frame)
        else:
            raise NotImplementedError("Only support anet and yc2, got {}".format(self.dset_name))
        return frame_to_second

    def _load_data(self, data_path):
        logger.info("Loading data from {}".format(data_path))
        raw_data = load_json(data_path)
        data = []
        for k, line in tqdm(raw_data.items()):
            line["name"] = k
            line["timestamps"] = line["timestamps"][:self.max_n_sen]
            line["sentences"] = line["sentences"][:self.max_n_sen]
            data.append(line)

        if self.recurrent:
            self.data = data
        else:  # non-recurrent single sentence
            single_sentence_data = []
            for d in data:
                num_sen = min(self.max_n_sen, len(d["sentences"]))
                single_sentence_data.extend([
                    {
                        "duration": d["duration"],
                        "name": d["name"],
                        "timestamp": d["timestamps"][idx],
                        "sentence": d["sentences"][idx]
                    } for idx in range(num_sen)])
            self.data = single_sentence_data

        logger.info("Loading complete! {} examples".format(len(self)))

    # ─────────────────────────────────────────────────────────────────────────
    # Feature-to-model conversion
    # ─────────────────────────────────────────────────────────────────────────

    def convert_example_to_features(self, example):
        """example single sentence
        {"name": str,
         "duration": float,
         "timestamp": [st(float), ed(float)],
         "sentence": str
        } or
        {"name": str,
         "duration": float,
         "timestamps": list([st(float), ed(float)]),
         "sentences": list(str)
        }
        """
        name = example["name"]
        video_name = name[2:] if self.dset_name == "anet" else name

        # Load C3D + flow concatenated: (100, 3072)
        video_feature = self._load_video_feature(video_name)

        if self.recurrent:
            num_sen = len(example["sentences"])
            single_video_features = []
            single_video_meta = []
            for clip_idx in range(num_sen):
                cur_data, cur_meta = self.clip_sentence_to_feature(
                    example["name"],
                    example["timestamps"][clip_idx],
                    example["sentences"][clip_idx],
                    video_feature,
                )
                single_video_features.append(cur_data)
                single_video_meta.append(cur_meta)
            return single_video_features, single_video_meta
        else:  # single sentence
            if self.untied:
                cur_data, cur_meta = self.clip_sentence_to_feature_untied(
                    example["name"],
                    example["timestamp"],
                    example["sentence"],
                    video_feature,
                )
            else:
                cur_data, cur_meta = self.clip_sentence_to_feature(
                    example["name"],
                    example["timestamp"],
                    example["sentence"],
                    video_feature,
                )
            return cur_data, cur_meta

    def clip_sentence_to_feature(self, name, timestamp, sentence, video_feature):
        """Make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]

        With C3D, every video has exactly max_v_len clips so no skimming or
        downsampling is required — the full feature array fills the video slots.

        Args:
            name:          str
            timestamp:     [float, float]  (kept for timestamp PE)
            sentence:      str
            video_feature: (100, 3072) float32 array (C3D + resampled flow)
        """
        feat, video_tokens, video_mask = self._load_video_feature_fixed(video_feature)

        # Passo 5: inject timestamp positional encoding
        feat = self._inject_timestamp_encoding(feat, timestamp, video_tokens)

        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)
        input_tokens = video_tokens + text_tokens

        input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens]
        input_labels = (
            [self.IGNORE] * len(video_tokens) +
            [self.IGNORE if m == 0 else tid
             for tid, m in zip(input_ids[-len(text_mask):], text_mask)][1:] +
            [self.IGNORE]
        )
        input_mask = video_mask + text_mask
        token_type_ids = [0] * self.max_v_len + [1] * self.max_t_len

        data = dict(
            name=name,
            input_tokens=input_tokens,
            input_ids=np.array(input_ids).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64),
            input_mask=np.array(input_mask).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
            video_feature=feat.astype(np.float32),
        )
        meta = dict(name=name, timestamp=timestamp, sentence=sentence)
        return data, meta

    def clip_sentence_to_feature_untied(self, name, timestamp, sentence, video_feature):
        """Make features for a single clip-sentence pair (untied mode).

        Args:
            name:          str
            timestamp:     [float, float]
            sentence:      str
            video_feature: (100, 3072) float32 array
        """
        feat, video_mask = self._load_video_feature_fixed_untied(video_feature)

        # Passo 5: timestamp encoding for untied mode
        n_valid = int(sum(video_mask))
        video_tokens_proxy = (
            [self.VID_TOKEN] * n_valid + [self.PAD_TOKEN] * (self.max_v_len - n_valid)
        )
        feat = self._inject_timestamp_encoding(
            np.pad(feat, ((0, 0), (0, 0))),  # no-op pad, just for API consistency
            timestamp,
            video_tokens_proxy,
        )

        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)
        text_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in text_tokens]
        text_labels = (
            [self.IGNORE if m == 0 else tid for tid, m in zip(text_ids, text_mask)][1:] +
            [self.IGNORE]
        )

        data = dict(
            name=name,
            text_tokens=text_tokens,
            text_ids=np.array(text_ids).astype(np.int64),
            text_mask=np.array(text_mask).astype(np.float32),
            text_labels=np.array(text_labels).astype(np.int64),
            video_feature=feat.astype(np.float32),
            video_mask=np.array(video_mask).astype(np.float32),
        )
        meta = dict(name=name, timestamp=timestamp, sentence=sentence)
        return data, meta

    # ─────────────────────────────────────────────────────────────────────────
    # Fixed-length video loading (no skimming, no timestamp indexing)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_video_feature_fixed(self, raw_feat: np.ndarray):
        """Pack a fixed-size video feature into the model input buffer.

        Because C3D features already have exactly 100 clips (== max_v_len - 2
        after reserving slots for [CLS] and [SEP]), no timestamp-based indexing
        or skimming is needed.  The layout is:

            [CLS] [VID]*98 [SEP]   (max_v_len = 100, so 98 VID slots)

        If a video has fewer than 98 clips (edge case), the remaining slots are
        zero-padded.  If it has more, only the first 98 are used.

        Args:
            raw_feat: (N, D) float32 array — typically (100, 3072)

        Returns:
            feat:         (max_v_len + max_t_len, D) zero-padded array
            video_tokens: list[str] of length max_v_len
            video_mask:   list[int] of length max_v_len  (1 = valid, 0 = pad)
        """
        max_v_l = self.max_v_len - 2  # slots for [VID] tokens (excl. CLS + SEP)
        D = raw_feat.shape[1]
        n_clips = min(len(raw_feat), max_v_l)

        feat = np.zeros((self.max_v_len + self.max_t_len, D), dtype=np.float32)
        feat[1:n_clips + 1] = raw_feat[:n_clips]  # slot 0 = CLS (zeros), then VID

        video_tokens = (
            [self.CLS_TOKEN] +
            [self.VID_TOKEN] * n_clips +
            [self.SEP_TOKEN] +
            [self.PAD_TOKEN] * (max_v_l - n_clips)
        )
        video_mask = [1] * (n_clips + 2) + [0] * (max_v_l - n_clips)

        return feat, video_tokens, video_mask

    def _load_video_feature_fixed_untied(self, raw_feat: np.ndarray):
        """Untied version: only [VID] tokens, padded to max_v_len.

        Args:
            raw_feat: (N, D) float32 array — typically (100, 3072)

        Returns:
            feat: (max_v_len, D)
            mask: list[int] of length max_v_len
        """
        max_v_l = self.max_v_len
        D = raw_feat.shape[1]
        n_clips = min(len(raw_feat), max_v_l)

        feat = np.zeros((max_v_l, D), dtype=np.float32)
        feat[:n_clips] = raw_feat[:n_clips]
        mask = [1] * n_clips + [0] * (max_v_l - n_clips)

        return feat, mask

    # ─────────────────────────────────────────────────────────────────────────
    # Passo 5: Timestamp positional encoding
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sinusoidal_pe(position: float, dim: int) -> np.ndarray:
        """Scalar sinusoidal encoding for a single normalised position in [0, 1]."""
        pe = np.zeros(dim, dtype=np.float32)
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[0::2] = np.sin(position * div_term)
        pe[1::2] = np.cos(position * div_term[: len(pe[1::2])])
        return pe

    def _inject_timestamp_encoding(
        self,
        feat: np.ndarray,
        timestamp: list,
        video_tokens: list,
    ) -> np.ndarray:
        """Add sinusoidal timestamp encodings to the video feature array.

        Each valid [VID] position receives an encoding that reflects its
        normalised position within the clip (linearly interpolated between the
        clip's start and end time).  CLS, SEP, and PAD slots are untouched.

        Args:
            feat:         (max_v_len + max_t_len, D) or (max_v_len, D) array
            timestamp:    [start_sec, end_sec]
            video_tokens: list of token strings

        Returns:
            feat with timestamp PE added (copy)
        """
        feat = feat.copy()
        dim = feat.shape[1]
        t_start, t_end = float(timestamp[0]), float(timestamp[1])
        duration = max(t_end - t_start, 1e-6)

        vid_positions = [i for i, tok in enumerate(video_tokens) if tok == self.VID_TOKEN]
        n_vid = len(vid_positions)
        for rank, pos_idx in enumerate(vid_positions):
            norm_pos = rank / max(n_vid - 1, 1)
            pe = self._sinusoidal_pe(norm_pos, dim)
            feat[pos_idx] += pe

        return feat

    # ─────────────────────────────────────────────────────────────────────────
    # Text tokenisation
    # ─────────────────────────────────────────────────────────────────────────

    def _tokenize_pad_sentence(self, sentence):
        """[BOS], [WORD1], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1.
        """
        max_t_len = self.max_t_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


# ─────────────────────────────────────────────────────────────────────────────
# Batch utilities (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_batch_inputs(batch, device, non_blocking=False):
    batch_inputs = dict()
    bsz = len(batch["name"])
    for k, v in batch.items():
        assert bsz == len(v), (bsz, k, v)
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        else:
            batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def caption_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    raw_batch_meta = [e[1] for e in batch]
    batch_meta = []
    for e in raw_batch_meta:
        cur_meta = dict(name=None, timestamp=[], gt_sentence=[])
        for d in e:
            cur_meta["name"] = d["name"]
            cur_meta["timestamp"].append(d["timestamp"])
            cur_meta["gt_sentence"].append(d["sentence"])
        batch_meta.append(cur_meta)

    batch = [e[0] for e in batch]
    max_n_sen = max([len(e) for e in batch])
    raw_step_sizes = []

    padded_batch = []
    padding_clip_sen_data = copy.deepcopy(batch[0][0])
    padding_clip_sen_data["input_labels"][:] = RecursiveCaptionDataset.IGNORE
    for ele in batch:
        cur_n_sen = len(ele)
        if cur_n_sen < max_n_sen:
            ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
        raw_step_sizes.append(cur_n_sen)
        padded_batch.append(ele)

    collated_step_batch = []
    for step_idx in range(max_n_sen):
        collated_step = step_collate([e[step_idx] for e in padded_batch])
        collated_step_batch.append(collated_step)
    return collated_step_batch, raw_step_sizes, batch_meta


def single_sentence_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    batch_meta = [{"name": e[1]["name"],
                   "timestamp": e[1]["timestamp"],
                   "gt_sentence": e[1]["sentence"]
                   } for e in batch]
    padded_batch = step_collate([e[0] for e in batch])
    return padded_batch, None, batch_meta