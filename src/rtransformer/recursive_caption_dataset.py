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
        - C3D features:  <c3d_feature_dir>/<video_name>.npy          shape (N, 2048)
        - Flow features: <flow_feature_dir>/<video_name>_bn.npy       shape (M, 1024)

    Flow is resampled from M → N clips via linear interpolation so both
    modalities share the same temporal resolution before concatenation.
    Final video_feature shape per clip: (N, 3072)  [2048 C3D + 1024 flow]

    Temporal segmentation:
        Each segment [t_start, t_end] is converted to clip indices via
        `_convert_to_feat_index_st_ed`, which maps wall-clock seconds to
        feature indices proportionally to the video duration.  If the
        resulting segment is longer than (max_v_len - 2), it is uniformly
        downsampled; if shorter, it is zero-padded.

    Timestamp positional encoding:
        A sinusoidal PE proportional to each clip's position *within the
        segment* is added on top of the clipped features, preserving fine-
        grained temporal ordering inside the window.
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
                 max_t_len, max_v_len, max_n_sen, mode="train", recurrent=True, untied=False, lang_feature_dir=None, sent_feature_dir=None,):
        self.dset_name = dset_name
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.data_dir = data_dir
        self.duration_file = duration_file
        self._load_duration()
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len
        self.max_n_sen = max_n_sen

        self.c3d_feature_dir = video_feature_dir
        self.flow_feature_dir = flow_feature_dir
        self.lang_feature_dir = lang_feature_dir
        self.sent_feature_dir = sent_feature_dir

        self.mode = mode
        self.recurrent = recurrent
        self.untied = untied
        assert not (self.recurrent and self.untied), "untied and recurrent cannot be True for both"

        self.data = None
        self.set_data_mode(mode=mode)
        self.missing_video_names = []
        self.fix_missing()

        self.num_sens = None

    def _load_sent_feature(self, name):
        if self.sent_feature_dir is None:
            return None

        path = os.path.join(
            self.sent_feature_dir,
            name + ".json"
        )

        if not os.path.exists(path):
            return None

        feat = load_json(path)
        if feat is None:
            return None
        return np.asarray(feat, dtype=np.float32)

    def _load_lang_feature(self, name):
        """Load CLIP language features for all segments of a video.

        Expected file: <lang_feature_dir>/<video_name>.npy
        Shape: (num_segments, max_v_len, clip_lang_dim)
        Returns None if the directory is not set or the file is missing.
        """
        if self.lang_feature_dir is None:
            return None
        path = os.path.join(self.lang_feature_dir, name + ".npy")
        if not os.path.exists(path):
            return None
        return np.load(path).astype(np.float32)

    def _load_duration(self):
        """Load video durations in seconds.

        Stored as self.duration[vid_name] = float seconds.
        Used by _convert_to_feat_index_st_ed to map timestamps → clip indices.
        """
        duration = {}
        if self.dset_name == "anet":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    duration[vid_name] = float(vid_dur)
            duration["_0CqozZun3U"] = 294.227  # known missing video in anet (8818 frames)
        elif self.dset_name == "yc2":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    duration[vid_name] = float(vid_dur)
        else:
            raise NotImplementedError("Only support anet and yc2, got {}".format(self.dset_name))
        self.duration = duration

    def _c3d_path(self, video_name: str) -> str:
        return os.path.join(self.c3d_feature_dir, "v_{}.npy".format(video_name))

    def _flow_path(self, video_name: str) -> str:
        return os.path.join(self.flow_feature_dir, "{}_bn.npy".format(video_name))

    @staticmethod
    def _resample_flow(flow: np.ndarray, target_len: int) -> np.ndarray:
        """Linearly resample flow from its original length to *target_len*.

        Args:
            flow:       (src_len, 1024) float array
            target_len: desired number of clips (matches C3D clip count)

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
        """Load and concatenate C3D + flow features for the full video.

        C3D  : (N, 2048) — used as-is
        Flow : (M, 1024) — resampled to (N, 1024)
        Output: (N, 3072) float32
        """
        c3d = np.load(self._c3d_path(video_name)).astype(np.float32)   # (N, 2048)
        #flow = np.load(self._flow_path(video_name))                     # (M, 1024)
        #flow_resampled = self._resample_flow(flow, target_len=c3d.shape[0])
        return c3d#np.concatenate([c3d, flow_resampled], axis=1)            # (N, 3072)

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len: int, timestamp: list, duration: float) -> tuple:
        """Convert wall-clock [t_start, t_end] (seconds) to clip index [st, ed].

        Maps proportionally: st = floor(t_start / duration * feat_len)
                             ed = ceil (t_end   / duration * feat_len)

        Both are clamped so that 0 <= st < ed <= feat_len - 1.

        Args:
            feat_len:  total number of clips in the video
            timestamp: [t_start, t_end] in seconds
            duration:  total video duration in seconds

        Returns:
            (st, ed) inclusive clip indices
        """
        st = int(math.floor((timestamp[0] / duration) * feat_len))
        ed = int(math.ceil((timestamp[1] / duration) * feat_len))
        ed = min(ed, feat_len - 1)
        st = min(st, ed - 1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed

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
            if video_name not in self.duration:
                self.missing_video_names.append(video_name)

            paths_to_check = [self._c3d_path(video_name)]
            for p in paths_to_check:
                if not os.path.exists(p):
                    self.missing_video_names.append(video_name)
            
        logger.info("Missing {} features (clips/sentences) from {} videos".format(
            len(self.missing_video_names), len(set(self.missing_video_names))))
        logger.info("Missing {}".format(set(self.missing_video_names)))
        if self.dset_name == "anet":
            self.data = [e for e in self.data if e["name"][2:] not in self.missing_video_names]
        else:
            self.data = [e for e in self.data if e["name"] not in self.missing_video_names]

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

        video_feature = self._load_video_feature(video_name)

        if self.recurrent:
            num_sen = len(example["sentences"])
            single_video_features = []
            single_video_meta = []

            sent_feat_all = self._load_sent_feature(video_name)
            lang_feat_all = self._load_lang_feature(video_name)  # (num_seg, max_v_len, D_lang) or None

            for clip_idx in range(num_sen):
                cur_data, cur_meta = self.clip_sentence_to_feature(
                    example["name"],
                    example["timestamps"][clip_idx],
                    example["sentences"][clip_idx],
                    video_feature,
                    clip_idx,
                )

                sent_feat = None
                if sent_feat_all is not None and clip_idx < len(sent_feat_all):
                    sent_feat = sent_feat_all[clip_idx]
                if sent_feat is not None:
                    cur_data["sent_feat"] = sent_feat.astype(np.float32)

                if lang_feat_all is not None and clip_idx < len(lang_feat_all):
                    lang_feat = lang_feat_all[clip_idx]          # (max_v_len, D_lang)
                    cur_data["lang_feature"] = lang_feat.astype(np.float32)
                    cur_data["lang_mask"] = cur_data["input_mask"].copy()
                else:
                    D_lang = lang_feat_all.shape[-1] if lang_feat_all is not None else 1
                    cur_data["lang_feature"] = np.zeros(
                        (self.max_v_len + self.max_t_len, D_lang), dtype=np.float32
                    )
                    cur_data["lang_mask"] = np.zeros_like(cur_data["input_mask"])

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
                    clip_idx,
                )
            else:
                cur_data, cur_meta = self.clip_sentence_to_feature(
                    example["name"],
                    example["timestamp"],
                    example["sentence"],
                    video_feature,
                    clip_idx,
                )
            return cur_data, cur_meta

    def clip_sentence_to_feature(self, name, timestamp, sentence, video_feature, clip_idx=None):
        """Make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]

        The video window is temporally indexed to [t_start, t_end] before
        padding/downsampling to max_v_len slots.

        Args:
            name:          str
            timestamp:     [float, float]  (seconds)
            sentence:      str
            video_feature: (N, 3072) float32 array — full video C3D + flow
        """
        video_name = name[2:] if self.dset_name == "anet" else name
        duration = self.duration[video_name]

        feat, video_tokens, video_mask = self._load_indexed_video_feature(
            video_feature, timestamp, duration
        )

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
            video_feature=feat.astype(np.float32)
            )
        meta = dict(name=name, timestamp=timestamp, sentence=sentence)
        return data, meta

    def clip_sentence_to_feature_untied(self, name, timestamp, sentence, video_feature, clip_idx=None):
        """Make features for a single clip-sentence pair (untied mode).

        Args:
            name:          str
            timestamp:     [float, float]  (seconds)
            sentence:      str
            video_feature: (N, 3072) float32 array — full video C3D + flow
        """
        video_name = name[2:] if self.dset_name == "anet" else name
        duration = self.duration[video_name]

        feat, video_mask = self._load_indexed_video_feature_untied(
            video_feature, timestamp, duration
        )

        n_valid = int(sum(video_mask))
        video_tokens_proxy = (
            [self.VID_TOKEN] * n_valid + [self.PAD_TOKEN] * (self.max_v_len - n_valid)
        )
        feat = self._inject_timestamp_encoding(feat, timestamp, video_tokens_proxy)

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
            video_mask=np.array(video_mask).astype(np.float32)
        )
        meta = dict(name=name, timestamp=timestamp, sentence=sentence)
        return data, meta

    def _load_indexed_video_feature(self, raw_feat: np.ndarray, timestamp: list, duration: float):
        """Slice the video to the segment [t_start, t_end] and pack into the
        model input buffer.

        Layout:  [CLS] [VID]*valid_l [SEP] [PAD]*(max_v_l - valid_l)

        If the segment contains more clips than max_v_l, uniform downsampling
        is applied so no temporal information is silently discarded.

        Args:
            raw_feat:  (N, D) float32 — full video feature (C3D + flow)
            timestamp: [t_start, t_end] in seconds
            duration:  total video duration in seconds

        Returns:
            feat:         (max_v_len + max_t_len, D) zero-padded array
            video_tokens: list[str] of length max_v_len
            video_mask:   list[int] of length max_v_len  (1 = valid, 0 = pad)
        """
        max_v_l = self.max_v_len - 2  # slots for [VID] tokens (excl. CLS + SEP)
        feat_len = len(raw_feat)
        D = raw_feat.shape[1]

        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, duration)
        indexed_feat_len = ed - st + 1

        feat = np.zeros((self.max_v_len + self.max_t_len, D), dtype=np.float32)

        if indexed_feat_len > max_v_l:
            downsample_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(int).tolist()
            feat[1:max_v_l + 1] = raw_feat[downsample_indices]
            valid_l = max_v_l
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            video_mask = [1] * (max_v_l + 2)
        else:
            valid_l = indexed_feat_len
            feat[1:valid_l + 1] = raw_feat[st:ed + 1]
            video_tokens = (
                [self.CLS_TOKEN] +
                [self.VID_TOKEN] * valid_l +
                [self.SEP_TOKEN] +
                [self.PAD_TOKEN] * (max_v_l - valid_l)
            )
            video_mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)

        return feat, video_tokens, video_mask

    def _load_indexed_video_feature_untied(self, raw_feat: np.ndarray, timestamp: list, duration: float):
        """Untied version: [VID]*valid_l [PAD]*(max_v_len - valid_l).

        Args:
            raw_feat:  (N, D) float32 — full video feature
            timestamp: [t_start, t_end] in seconds
            duration:  total video duration in seconds

        Returns:
            feat: (max_v_len, D)
            mask: list[int] of length max_v_len
        """
        max_v_l = self.max_v_len
        feat_len = len(raw_feat)
        D = raw_feat.shape[1]

        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, duration)
        indexed_feat_len = ed - st + 1

        if indexed_feat_len > max_v_l:
            downsample_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(int).tolist()
            feat = raw_feat[downsample_indices].astype(np.float32)
            mask = [1] * max_v_l
        else:
            feat = np.zeros((max_v_l, D), dtype=np.float32)
            valid_l = indexed_feat_len
            feat[:valid_l] = raw_feat[st:ed + 1]
            mask = [1] * valid_l + [0] * (max_v_l - valid_l)

        return feat, mask

    @staticmethod
    def _sinusoidal_pe(position: float, dim: int) -> np.ndarray:
        """Scalar sinusoidal encoding for a single normalised position in [0, 1]."""
        pe = np.zeros(dim, dtype=np.float32)
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[0::2] = np.sin(position * div_term)
        pe[1::2] = np.cos(position * div_term[:len(pe[1::2])])
        return pe

    def _inject_timestamp_encoding(
        self,
        feat: np.ndarray,
        timestamp: list,
        video_tokens: list,
    ) -> np.ndarray:
        """Add sinusoidal timestamp encodings to the video feature array.

        Each valid [VID] position receives an encoding that reflects its
        normalised rank *within the segment window* (0 → first clip of the
        segment, 1 → last clip of the segment).  This complements the
        temporal slicing: slicing places the right clips in the buffer;
        this PE preserves their intra-segment ordering for the attention
        mechanism.

        Args:
            feat:         (max_v_len + max_t_len, D) or (max_v_len, D) array
            timestamp:    [start_sec, end_sec]  (kept for potential future use)
            video_tokens: list of token strings

        Returns:
            feat with timestamp PE added (copy)
        """
        feat = feat.copy()
        dim = feat.shape[1]

        vid_positions = [i for i, tok in enumerate(video_tokens) if tok == self.VID_TOKEN]
        n_vid = len(vid_positions)
        for rank, pos_idx in enumerate(vid_positions):
            norm_pos = rank / max(n_vid - 1, 1)
            pe = self._sinusoidal_pe(norm_pos, dim)
            feat[pos_idx] += pe

        return feat

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