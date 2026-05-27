"""
Recursive Transformer for dense video captioning.

Classes exported:
    LabelSmoothingLoss       – training loss with label smoothing
    RecursiveTransformer     – main recurrent captioning model
    base_config              – default hyper-parameter dict

Internal building blocks (not intended for external use):
    PositionEncoding, BertLayerNorm, BertSelfAttention, BertSelfOutput,
    BertAttention, BertIntermediate, BertOutput, BertLayerWithMemory,
    BertEncoderWithMemory, BertEmbeddingsWithVideo,
    MemoryInitializer, MemoryUpdater,
    BertPredictionHeadTransform, BertLMPredictionHead
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class LabelSmoothingLoss(nn.Module):
    """KL-divergence loss with label smoothing.

    Positions where ``target == ignore_index`` are excluded from the loss.
    """

    def __init__(
        self,
        label_smoothing: float,
        tgt_vocab_size: int,
        ignore_index: int = -100,
    ) -> None:
        if not 0.0 < label_smoothing <= 1.0:
            raise ValueError(f"label_smoothing must be in (0, 1], got {label_smoothing}")
        super().__init__()

        self.ignore_index = ignore_index
        self.confidence = 1.0 - label_smoothing
        smoothing_value = label_smoothing / (tgt_vocab_size - 1)

        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # Register as buffer so it moves with .to(device) / .cuda()
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: (batch_size, vocab_size) raw logits
            target: (batch_size,) ground-truth indices; ``ignore_index`` positions
                    are skipped
        """
        valid = target != self.ignore_index
        target = target[valid]
        output = self.log_softmax(output[valid])

        model_prob: torch.Tensor = self.one_hot.expand(target.size(0), -1).clone()
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016)."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionEncoding(nn.Module):
    """Sinusoidal positional encoding added to the last two dimensions of the input."""

    def __init__(self, n_filters: int = 128, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, n_filters)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_filters, 2, dtype=torch.float)
            * -(math.log(10000.0) / n_filters)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, L, D)
        Returns:
            (*, L, D) with positional encodings added
        """
        pe = self.pe[: x.size(-2)]  # (L, D)
        # Broadcast over any leading batch dimensions
        for _ in range(x.dim() - 2):
            pe = pe.unsqueeze(0)
        return x + pe


# ---------------------------------------------------------------------------
# Layer normalisation (TF-style: epsilon inside sqrt)
# ---------------------------------------------------------------------------

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class BertSelfAttention(nn.Module):
    def __init__(self, config: edict) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(N, L, D) -> (N, n_heads, L, head_size)"""
        N, L, _ = x.shape
        x = x.view(N, L, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_states: (N, Lq, D)
            key_states:   (N, L,  D)
            value_states: (N, L,  D)
            attention_mask: (N, Lq, L) with 1 = attend, 0 = mask

        Returns:
            (N, Lq, D)
        """
        # attention_mask: (N, 1, Lq, L), additive mask (-10000 for masked positions)
        additive_mask = (1.0 - attention_mask.unsqueeze(1)) * -10000.0

        q = self._split_heads(self.query(query_states))  # (N, nh, Lq, dh)
        k = self._split_heads(self.key(key_states))      # (N, nh, L,  dh)
        v = self._split_heads(self.value(value_states))  # (N, nh, L,  dh)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        scores = scores + additive_mask

        probs = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probs, v)  # (N, nh, Lq, dh)

        context = context.permute(0, 2, 1, 3).contiguous()
        N, Lq = context.shape[:2]
        return context.view(N, Lq, self.all_head_size)


class BertSelfOutput(nn.Module):
    def __init__(self, config: edict) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(residual + self.dropout(self.dense(hidden_states)))


class BertAttention(nn.Module):
    def __init__(self, config: edict) -> None:
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Self-attention with residual connection.

        Args:
            x:              (N, L, D)
            attention_mask: (N, L, L)  – causal / pad mask (1 = attend)
        """
        attn_out = self.self(x, x, x, attention_mask)
        return self.output(attn_out, x)


class BertIntermediate(nn.Module):
    def __init__(self, config: edict) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu(self.dense(x))


class BertOutput(nn.Module):
    def __init__(self, config: edict) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(residual + self.dropout(self.dense(hidden_states)))


# ---------------------------------------------------------------------------
# Attention mask helpers
# ---------------------------------------------------------------------------

def make_shifted_mask(
    input_mask: torch.Tensor,
    max_v_len: int,
    max_t_len: int,
    memory_len: int = 0,
) -> torch.Tensor:
    """Build the causal attention mask for the joint (memory +) video + text sequence.

    ``input_mask`` has length ``max_v_len + max_t_len + memory_len``.
    The query dimension spans the video + text positions (``max_v_len + max_t_len``);
    the key dimension spans all positions including any prepended memory tokens.

    Attend rules:
    - All query rows can see all memory and video key positions (full attention).
    - Text query rows additionally see text key positions causally (lower-triangular).

    Args:
        input_mask: (N, max_v_len + max_t_len + memory_len)  — 1 = valid token.
                    When memory_len == 0 this is just (N, max_v_len + max_t_len).
                    When memory_len > 0 the caller must have prepended memory
                    validity bits to input_mask before calling.
        max_v_len:  video segment length (query dimension)
        max_t_len:  text segment length  (query dimension)
        memory_len: number of prepended memory tokens (key dimension only)

    Returns:
        (N, max_v_len + max_t_len, max_v_len + max_t_len + memory_len)
        with 1 = attend, 0 = mask
    """
    bsz, seq_len = input_mask.shape
    assert max_v_len + max_t_len + memory_len == seq_len, (
        f"input_mask length {seq_len} != "
        f"max_v_len({max_v_len}) + max_t_len({max_t_len}) + memory_len({memory_len})"
    )
    query_len = max_v_len + max_t_len

    # All query positions can see memory + video keys
    shifted = input_mask.new_zeros(bsz, query_len, seq_len)
    shifted[:, :, : memory_len + max_v_len] = 1

    # Text query rows get causal access to text key columns
    causal = torch.tril(input_mask.new_ones(max_t_len, max_t_len))
    shifted[:, max_v_len:, memory_len + max_v_len:] = causal

    return shifted


def make_pad_shifted_mask(
    input_mask: torch.Tensor,
    max_v_len: int,
    max_t_len: int,
    memory_len: int = 0,
) -> torch.Tensor:
    """Shifted mask further zeroed at padding positions in the key dimension.

    When ``memory_len == 0`` the key padding comes directly from ``input_mask``.
    When ``memory_len > 0`` the caller is expected to have prepended the memory
    validity bits to ``input_mask`` already (memory slots are always valid).

    Args:
        input_mask: (N, memory_len + max_v_len + max_t_len)
    Returns:
        (N, max_v_len + max_t_len, memory_len + max_v_len + max_t_len)
    """
    shifted = make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len)
    # Zero columns that correspond to padding in the key sequence
    return shifted * input_mask.unsqueeze(1)


def make_video_only_mask(input_mask: torch.Tensor, max_v_len: int) -> torch.Tensor:
    """Return a copy of input_mask with text positions zeroed out.

    Used to restrict memory initialisation to the video context.
    """
    video_mask = input_mask.clone()
    video_mask[:, max_v_len:] = 0
    return video_mask


# ---------------------------------------------------------------------------
# Memory mechanism (expanded capacity)
# ---------------------------------------------------------------------------

class MemoryInitializer(nn.Module):
    """Initialise recurrent memory cells from the video context.

    Improvement over the original design: instead of a single global pooling
    vector repeated ``n_memory_cells`` times, we learn ``n_memory_cells``
    independent linear projections over the pooled context, giving each cell
    a distinct initialisation subspace.
    """

    def __init__(self, config: edict) -> None:
        super().__init__()
        self.n_memory_cells = config.n_memory_cells

        # One projection head per memory cell – enables richer initialisation
        self.cell_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                BertLayerNorm(config.hidden_size),
                nn.Dropout(config.memory_dropout_prob),
            )
            for _ in range(config.n_memory_cells)
        ])

    def forward(
        self, input_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_states:   (N, L, D)
            attention_mask: (N, L)   1 = valid

        Returns:
            (N, n_memory_cells, D)
        """
        # Masked average pooling over valid positions
        denom = attention_mask.sum(1, keepdim=True).clamp(min=1)     # (N, 1)
        pooled = (input_states * attention_mask.unsqueeze(-1)).sum(1) # (N, D)
        pooled = pooled / denom                                        # (N, D)

        # Apply one independent projection per cell
        cells = [proj(pooled).unsqueeze(1) for proj in self.cell_projections]
        return torch.cat(cells, dim=1)  # (N, n_memory_cells, D)


class MemoryUpdater(nn.Module):
    """GRU-style recurrent update for the memory cells.

    ``s_t``  – attention-based read from the current input context
    ``c_t``  – candidate new content  (tanh gate)
    ``z_t``  – interpolation gate     (sigmoid)
    ``m_t``  – updated memory         = (1 - z_t) * c_t + z_t * prev_m
    """

    def __init__(self, config: edict) -> None:
        super().__init__()
        self.memory_update_attention = BertSelfAttention(config)

        # Content gate
        self.mc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sc = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        # Interpolation gate
        self.mz = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sz = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        prev_m: torch.Tensor,
        input_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prev_m:         (N, M, D)  previous memory
            input_states:   (N, L, D)  current step hidden states
            attention_mask: (N, L)     1 = valid position

        Returns:
            updated_memory: (N, M, D)
        """
        n_cells = prev_m.size(1)
        # Each memory cell attends over all valid input positions
        update_mask = attention_mask.unsqueeze(1).expand(-1, n_cells, -1)  # (N, M, L)
        s_t = self.memory_update_attention(prev_m, input_states, input_states, update_mask)

        c_t = torch.tanh(self.mc(prev_m) + self.sc(s_t))
        z_t = torch.sigmoid(self.mz(prev_m) + self.sz(s_t))
        return (1.0 - z_t) * c_t + z_t * prev_m


# ---------------------------------------------------------------------------
# Transformer layer with memory-augmented attention
# ---------------------------------------------------------------------------

class BertLayerWithMemory(nn.Module):
    """Single transformer layer that reads from and writes to a memory bank.

    Processing order per step:
        1. Causal self-attention over (video + text) context
        2. Initialise memory from video if this is the first step
        3. Update memory cells via ``MemoryUpdater``
        4. Memory-augmented cross-attention: current hidden states attend to
           (prev_memory ∥ current_context)
        5. Feed-forward + residual
    """

    def __init__(self, config: edict) -> None:
        super().__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.memory_initializer = MemoryInitializer(config)
        self.memory_updater = MemoryUpdater(config)
        self.memory_augmented_attention = BertSelfAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_projection = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output = BertOutput(config)

    def forward(
        self,
        prev_m: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prev_m:         (N, M, D) or None on the first recurrent step
            hidden_states:  (N, L, D)
            attention_mask: (N, L)    1 = valid

        Returns:
            updated_m:     (N, M, D)
            layer_output:  (N, L, D)
        """
        max_v_len = self.config.max_v_len
        max_t_len = self.config.max_t_len

        # 1. Causal self-attention (video tokens see all; text tokens are causal)
        shifted_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)
        attention_output = self.attention(hidden_states, shifted_mask)

        # 2. Intermediate projection (feeds both memory and output paths)
        intermediate_output = self.hidden_intermediate(attention_output)

        # 3. Initialise or update memory
        if prev_m is None:
            init_mask = make_video_only_mask(attention_mask, max_v_len)
            prev_m = self.memory_initializer(intermediate_output, init_mask)

        updated_m = self.memory_updater(prev_m, intermediate_output, attention_mask)

        # 4. Memory-augmented attention:
        #    query  = current intermediate states
        #    key/value = concat(prev_memory, intermediate_states)
        bsz, n_cells = prev_m.shape[:2]
        concat_mh = torch.cat([prev_m, intermediate_output], dim=1)  # (N, M+L, D)

        # Build attention mask that includes the memory slots (always visible)
        mem_ones = attention_mask.new_ones(bsz, n_cells)
        raw_mem_mask = torch.cat([mem_ones, attention_mask], dim=1)  # (N, M+L)
        mem_attn_mask = make_pad_shifted_mask(
            raw_mem_mask, max_v_len, max_t_len, memory_len=n_cells
        )
        mem_attn_out = self.memory_augmented_attention(
            intermediate_output, concat_mh, concat_mh, mem_attn_mask
        )

        # 5. Project back to hidden_size and apply residual + layer norm
        mem_attn_out = self.memory_projection(mem_attn_out)
        layer_output = self.output(mem_attn_out, attention_output)

        return updated_m, layer_output


class BertEncoderWithMemory(nn.Module):
    """Stack of ``BertLayerWithMemory`` layers sharing a single memory bank."""

    def __init__(self, config: edict) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [BertLayerWithMemory(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        prev_ms: list[Optional[torch.Tensor]],
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            prev_ms:        list of length num_hidden_layers; each element is
                            (N, M, D) or None (first step)
            hidden_states:  (N, L, D)
            attention_mask: (N, L)

        Returns:
            updated_ms:        list[(N, M, D)] * num_hidden_layers
            all_encoder_layers: list[(N, L, D)] – one per layer (or just the last)
        """
        all_encoder_layers: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            prev_ms[i], hidden_states = layer(prev_ms[i], hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return prev_ms, all_encoder_layers


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class BertEmbeddingsWithVideo(nn.Module):
    """Fuse word, video and token-type embeddings into a single sequence.

    The joint sequence is laid out as ``[video tokens | text tokens]``.
    Video feature vectors are projected into the model dimension and added to
    the word embeddings at each position, allowing the model to see both
    modalities in a single unified sequence.
    """

    def __init__(self, config: edict, add_position_embeddings: bool = True) -> None:
        super().__init__()
        self.add_position_embeddings = add_position_embeddings

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.word_vec_size, padding_idx=0
        )
        self.word_fc = nn.Sequential(
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(inplace=True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.video_embeddings = nn.Sequential(
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(inplace=True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        if self.add_position_embeddings:
            self.position_embeddings = PositionEncoding(
                n_filters=config.hidden_size,
                max_len=config.max_position_embeddings,
            )

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_pretrained_embedding(
        self, pretrained_embedding: torch.Tensor, freeze: bool = True
    ) -> None:
        """Replace the word embedding table with pre-trained vectors (e.g. GloVe)."""
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape, (
            "Pre-trained embedding shape must match the current embedding table."
        )
        self.word_embeddings = nn.Embedding.from_pretrained(
            pretrained_embedding,
            freeze=freeze,
            padding_idx=self.word_embeddings.padding_idx,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        video_features: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (N, L)
            video_features: (N, L, D_v)  – zeros at text positions
            token_type_ids: (N, L)        – 0 for video, 1 for text

        Returns:
            (N, L, D)
        """
        word_emb = self.word_fc(self.word_embeddings(input_ids))
        vid_emb = self.video_embeddings(video_features)
        type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + vid_emb + type_emb
        if self.add_position_embeddings:
            embeddings = self.position_embeddings(embeddings)
        return self.dropout(self.LayerNorm(embeddings))


# ---------------------------------------------------------------------------
# Prediction head
# ---------------------------------------------------------------------------

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config: edict) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(gelu(self.dense(hidden_states)))


class BertLMPredictionHead(nn.Module):
    """Language-model head: hidden states -> vocabulary logits.

    Optionally shares weights with the input word embedding table
    (``config.share_wd_cls_weight``).
    """

    def __init__(
        self,
        config: edict,
        bert_model_embedding_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        if config.share_wd_cls_weight:
            if bert_model_embedding_weights is None:
                raise ValueError(
                    "bert_model_embedding_weights must be provided when "
                    "share_wd_cls_weight is True."
                )
            if config.hidden_size != bert_model_embedding_weights.size(1):
                raise ValueError(
                    "hidden_size must equal word_vec_size when sharing "
                    "embedding and classifier weights."
                )
            vocab_size, emb_dim = bert_model_embedding_weights.shape
            self.decoder = nn.Linear(emb_dim, vocab_size, bias=False)
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(N, L, D) -> (N, L, vocab_size)"""
        return self.decoder(self.transform(hidden_states)) + self.bias


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class RecursiveTransformer(nn.Module):
    """Recurrent video captioning model.

    At each step the model receives one video clip + partial caption and
    propagates a memory state to the next step, enabling the generated
    captions to be temporally coherent across a whole video.
    """

    def __init__(self, config: edict) -> None:
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddingsWithVideo(config, add_position_embeddings=True)
        self.encoder = BertEncoderWithMemory(config)

        decoder_weight = (
            self.embeddings.word_embeddings.weight
            if config.share_wd_cls_weight
            else None
        )
        self.decoder = BertLMPredictionHead(config, decoder_weight)

        if getattr(config, "label_smoothing", 0.0) > 0.0:
            self.loss_func: nn.Module = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=-1
            )
        else:
            self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, BertLayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    # ------------------------------------------------------------------
    # Core forward (single recurrent step)
    # ------------------------------------------------------------------

    def forward_step(
        self,
        prev_ms: list[Optional[torch.Tensor]],
        input_ids: torch.Tensor,
        video_features: torch.Tensor,
        input_masks: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Single recurrent step.

        Args:
            prev_ms:        list[(N, M, D)] * num_hidden_layers, or [None, …] at step 0
            input_ids:      (N, L)
            video_features: (N, L, D_v)
            input_masks:    (N, L)   1 = valid
            token_type_ids: (N, L)

        Returns:
            prev_ms:              updated memory list
            encoded_layer_outputs: list of (N, L, D) per encoder layer
            prediction_scores:    (N, L, vocab_size)
        """
        embeddings = self.embeddings(input_ids, video_features, token_type_ids)
        prev_ms, encoded_layer_outputs = self.encoder(
            prev_ms, embeddings, input_masks, output_all_encoded_layers=False
        )
        prediction_scores = self.decoder(encoded_layer_outputs[-1])
        return prev_ms, encoded_layer_outputs, prediction_scores

    # ------------------------------------------------------------------
    # Training / evaluation forward (unrolls over all caption steps)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids_list: list[torch.Tensor],
        video_features_list: list[torch.Tensor],
        input_masks_list: list[torch.Tensor],
        token_type_ids_list: list[torch.Tensor],
        input_labels_list: Optional[list[torch.Tensor]],
        return_memory: bool = False,
    ):
        """Unroll the recurrent model over all caption steps.

        Args:
            input_ids_list:      [(N, L)] * step_size
            video_features_list: [(N, L, D_v)] * step_size
            input_masks_list:    [(N, L)] * step_size
            token_type_ids_list: [(N, L)] * step_size
            input_labels_list:   [(N, L)] * step_size  or None when return_memory=True
            return_memory:       if True, return accumulated memory states instead
                                 of loss (used for analysis)

        Returns:
            If ``return_memory`` is False:
                (caption_loss, prediction_scores_list)
            If ``return_memory`` is True:
                memory_list – list of memory states per step per layer
        """
        prev_ms: list[Optional[torch.Tensor]] = [None] * self.config.num_hidden_layers
        step_size = len(input_ids_list)

        memory_list: list = []
        prediction_scores_list: list[torch.Tensor] = []

        for idx in range(step_size):
            prev_ms, _, prediction_scores = self.forward_step(
                prev_ms,
                input_ids_list[idx],
                video_features_list[idx],
                input_masks_list[idx],
                token_type_ids_list[idx],
            )
            memory_list.append(prev_ms)
            prediction_scores_list.append(prediction_scores)

        if return_memory:
            return memory_list

        if input_labels_list is None:
            raise ValueError("input_labels_list is required when return_memory=False")

        caption_loss = sum(
            self.loss_func(
                prediction_scores_list[i].view(-1, self.config.vocab_size),
                input_labels_list[i].view(-1),
            )
            for i in range(step_size)
        )
        return caption_loss, prediction_scores_list


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

base_config = edict(
    hidden_size=768,
    vocab_size=None,               # populated from word2idx at runtime
    video_feature_size=2048,
    max_position_embeddings=None,  # populated from max_seq_len at runtime
    max_v_len=100,
    max_t_len=30,
    n_memory_cells=10,             # increase for richer temporal context
    type_vocab_size=2,
    layer_norm_eps=1e-12,
    hidden_dropout_prob=0.1,
    num_hidden_layers=2,
    attention_probs_dropout_prob=0.1,
    intermediate_size=768,
    num_attention_heads=12,
    memory_dropout_prob=0.1,
    word_vec_size=300,
    share_wd_cls_weight=False,
    label_smoothing=0.1,
    initializer_range=0.02,
)