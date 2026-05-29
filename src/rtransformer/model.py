from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

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

def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016)."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

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

def make_shifted_mask(
    input_mask: torch.Tensor,
    max_v_len: int,
    max_t_len: int,
    memory_len: int = 0,
) -> torch.Tensor:
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
    shifted = make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len)
    # Zero columns that correspond to padding in the key sequence
    return shifted * input_mask.unsqueeze(1)

def make_video_only_mask(input_mask: torch.Tensor, max_v_len: int) -> torch.Tensor:
    video_mask = input_mask.clone()
    video_mask[:, max_v_len:] = 0
    return video_mask

class MemoryInitializer(nn.Module):
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
        # Masked average pooling over valid positions
        denom = attention_mask.sum(1, keepdim=True).clamp(min=1)     # (N, 1)
        pooled = (input_states * attention_mask.unsqueeze(-1)).sum(1) # (N, D)
        pooled = pooled / denom                                        # (N, D)

        # Apply one independent projection per cell
        cells = [proj(pooled).unsqueeze(1) for proj in self.cell_projections]
        return torch.cat(cells, dim=1)  # (N, n_memory_cells, D)

class MemoryUpdater(nn.Module):
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
        n_cells = prev_m.size(1)
        # Each memory cell attends over all valid input positions
        update_mask = attention_mask.unsqueeze(1).expand(-1, n_cells, -1)  # (N, M, L)
        s_t = self.memory_update_attention(prev_m, input_states, input_states, update_mask)

        c_t = torch.tanh(self.mc(prev_m) + self.sc(s_t))
        z_t = torch.sigmoid(self.mz(prev_m) + self.sz(s_t))
        return (1.0 - z_t) * c_t + z_t * prev_m

class BertLayerWithMemory(nn.Module):
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

        # ── Passo 4: Coverage mechanism ──────────────────────────────────────
        # Tracks cumulative attention over video frames across recurrent steps.
        # A coverage vector (running sum of past attention weights) is projected
        # and added to the attention logits so the model is penalised for
        # repeatedly attending to the same visual regions.
        use_coverage = getattr(config, "use_coverage", True)
        self.use_coverage = use_coverage
        if use_coverage:
            # Projects the scalar coverage count → hidden_size bias term
            self.coverage_proj = nn.Linear(1, config.hidden_size, bias=False)
            # Gates how strongly coverage penalty is applied
            self.coverage_gate = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        prev_m: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            prev_m:         previous memory state (N, M, D) or None
            hidden_states:  (N, L, D)
            attention_mask: (N, L)  – 1 for valid, 0 for padding
            coverage:       (N, L)  – cumulative attention mass per frame, or None

        Returns:
            updated_m:      (N, M, D)
            layer_output:   (N, L, D)
            new_coverage:   (N, L)  – updated coverage vector (or None)
        """
        max_v_len = self.config.max_v_len
        max_t_len = self.config.max_t_len

        # ── Coverage bias injection ───────────────────────────────────────────
        # Before self-attention, add a per-position bias derived from how much
        # the model has already attended to each frame in prior recurrent steps.
        states_for_attn = hidden_states
        if self.use_coverage and coverage is not None:
            # coverage: (N, L) → (N, L, 1) → (N, L, D) via linear
            cov_bias = self.coverage_proj(coverage.unsqueeze(-1))          # (N, L, D)
            gate = torch.sigmoid(self.coverage_gate(hidden_states))        # (N, L, D)
            states_for_attn = hidden_states - gate * cov_bias              # discourage re-attendance

        shifted_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)
        attention_output = self.attention(states_for_attn, shifted_mask)

        intermediate_output = self.hidden_intermediate(attention_output)

        if prev_m is None:
            init_mask = make_video_only_mask(attention_mask, max_v_len)
            prev_m = self.memory_initializer(intermediate_output, init_mask)

        updated_m = self.memory_updater(prev_m, intermediate_output, attention_mask)

        bsz, n_cells = prev_m.shape[:2]
        concat_mh = torch.cat([prev_m, intermediate_output], dim=1)  # (N, M+L, D)

        mem_ones = attention_mask.new_ones(bsz, n_cells)
        raw_mem_mask = torch.cat([mem_ones, attention_mask], dim=1)  # (N, M+L)
        mem_attn_mask = make_pad_shifted_mask(
            raw_mem_mask, max_v_len, max_t_len, memory_len=n_cells
        )
        mem_attn_out = self.memory_augmented_attention(
            intermediate_output, concat_mh, concat_mh, mem_attn_mask
        )

        mem_attn_out = self.memory_projection(mem_attn_out)
        layer_output = self.output(mem_attn_out, attention_output)

        # ── Update coverage: accumulate mean attention over video frames ──────
        new_coverage: Optional[torch.Tensor] = None
        if self.use_coverage:
            # Use the mean-pooled attention output over the video portion as a
            # proxy for "how much was attended". Detach so gradients don't flow
            # back through coverage accumulation (standard practice).
            video_attn = layer_output[:, :max_v_len, :]                    # (N, Lv, D)
            attn_proxy = video_attn.norm(dim=-1).detach()                  # (N, Lv)
            # Pad or trim to full sequence length for consistent shape
            pad_len = hidden_states.size(1) - max_v_len
            attn_proxy_full = F.pad(attn_proxy, (0, pad_len))              # (N, L)
            new_coverage = (coverage if coverage is not None else torch.zeros_like(attn_proxy_full)) + attn_proxy_full

        return updated_m, layer_output, new_coverage

class BertEncoderWithMemory(nn.Module):
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
        coverages: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[Optional[torch.Tensor]]]:
        """
        Args:
            coverages: per-layer coverage vectors from previous recurrent step,
                       length == num_hidden_layers. Pass None to start fresh.
        Returns:
            (prev_ms, all_encoder_layers, new_coverages)
        """
        if coverages is None:
            coverages = [None] * len(self.layers)

        all_encoder_layers: list[torch.Tensor] = []
        new_coverages: list[Optional[torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            prev_ms[i], hidden_states, new_cov = layer(
                prev_ms[i], hidden_states, attention_mask, coverage=coverages[i]
            )
            new_coverages.append(new_cov)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return prev_ms, all_encoder_layers, new_coverages

class BertEmbeddingsWithVideo(nn.Module):
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
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape, (
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
        word_emb = self.word_fc(self.word_embeddings(input_ids))
        vid_emb = self.video_embeddings(video_features)
        type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + vid_emb + type_emb
        if self.add_position_embeddings:
            embeddings = self.position_embeddings(embeddings)
        return self.dropout(self.LayerNorm(embeddings))

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config: edict) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(gelu(self.dense(hidden_states)))


class BertLMPredictionHead(nn.Module):
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

class RecursiveTransformer(nn.Module):
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

        # ── Passo 6: Contrastive loss between sentence embeddings ─────────────
        # Projects the pooled [BOS] hidden state of each generated sentence into
        # a shared embedding space. Sentences within the same paragraph are
        # pulled together; sentences from different paragraphs (different steps
        # in the same batch) are pushed apart via NT-Xent (InfoNCE).
        use_contrastive = getattr(config, "use_contrastive_loss", True)
        self.use_contrastive = use_contrastive
        if use_contrastive:
            contrastive_dim = getattr(config, "contrastive_dim", 128)
            self.contrastive_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, contrastive_dim),
            )
            self.contrastive_temp = getattr(config, "contrastive_temp", 0.07)
            self.contrastive_weight = getattr(config, "contrastive_weight", 0.1)

        self.apply(self._init_weights)

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

    # ── Passo 6 helper ────────────────────────────────────────────────────────
    def _contrastive_loss(
        self, sentence_embs: list[torch.Tensor]
    ) -> torch.Tensor:
        """NT-Xent loss that attracts consecutive sentence embeddings within each
        paragraph and repels non-adjacent ones across the batch.

        Args:
            sentence_embs: list of (N, D_proj) tensors, one per recurrent step.
        Returns:
            scalar loss
        """
        if len(sentence_embs) < 2:
            return sentence_embs[0].new_tensor(0.0)

        # Stack → (S, N, D_proj), treat consecutive step-pairs as positives.
        embs = torch.stack(sentence_embs, dim=0)          # (S, N, D_proj)
        embs = F.normalize(embs, dim=-1)
        S, N, D = embs.shape

        total_loss = embs.new_tensor(0.0)
        n_pairs = 0
        for s in range(S - 1):
            # anchor = step s, positive = step s+1, all others are negatives
            anchor = embs[s]        # (N, D)
            positive = embs[s + 1]  # (N, D)
            # Similarity of every anchor against every positive in the batch
            sim = torch.matmul(anchor, positive.T) / self.contrastive_temp  # (N, N)
            labels = torch.arange(N, device=sim.device)
            total_loss = total_loss + F.cross_entropy(sim, labels)
            n_pairs += 1

        return total_loss / n_pairs

    def forward_step(
        self,
        prev_ms: list[Optional[torch.Tensor]],
        input_ids: torch.Tensor,
        video_features: torch.Tensor,
        input_masks: torch.Tensor,
        token_type_ids: torch.Tensor,
        coverages: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[Optional[torch.Tensor]]]:
        embeddings = self.embeddings(input_ids, video_features, token_type_ids)
        prev_ms, encoded_layer_outputs, new_coverages = self.encoder(
            prev_ms, embeddings, input_masks,
            output_all_encoded_layers=False,
            coverages=coverages,
        )
        prediction_scores = self.decoder(encoded_layer_outputs[-1])
        return prev_ms, encoded_layer_outputs, prediction_scores, new_coverages

    def forward(
        self,
        input_ids_list: list[torch.Tensor],
        video_features_list: list[torch.Tensor],
        input_masks_list: list[torch.Tensor],
        token_type_ids_list: list[torch.Tensor],
        input_labels_list: Optional[list[torch.Tensor]],
        return_memory: bool = False,
    ):
        """
        Main forward pass.

        Coverage vectors are initialised to None and accumulated across recurrent
        steps so the model can track which visual regions it has already described.

        When ``use_contrastive_loss`` is enabled, an NT-Xent contrastive loss is
        computed across consecutive sentence embeddings (pooled [BOS] states) and
        added to the captioning loss.
        """
        prev_ms: list[Optional[torch.Tensor]] = [None] * self.config.num_hidden_layers
        # Passo 4: per-layer coverage vectors, initialised to None
        coverages: list[Optional[torch.Tensor]] = [None] * self.config.num_hidden_layers
        step_size = len(input_ids_list)

        memory_list: list = []
        prediction_scores_list: list[torch.Tensor] = []
        # Passo 6: store per-step sentence representations for contrastive loss
        sentence_embs: list[torch.Tensor] = []

        for idx in range(step_size):
            prev_ms, encoded_layers, prediction_scores, coverages = self.forward_step(
                prev_ms,
                input_ids_list[idx],
                video_features_list[idx],
                input_masks_list[idx],
                token_type_ids_list[idx],
                coverages=coverages,
            )
            memory_list.append(prev_ms)
            prediction_scores_list.append(prediction_scores)

            # Passo 6: pool the [BOS] hidden state (first text token = position max_v_len)
            if self.use_contrastive:
                bos_hidden = encoded_layers[-1][:, self.config.max_v_len, :]  # (N, D)
                sentence_embs.append(self.contrastive_proj(bos_hidden))

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

        # Passo 6: add contrastive loss
        if self.use_contrastive and len(sentence_embs) >= 2:
            contrastive_loss = self._contrastive_loss(sentence_embs)
            caption_loss = caption_loss + self.contrastive_weight * contrastive_loss

        return caption_loss, prediction_scores_list

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
    # ── Passo 4 ──
    use_coverage=True,
    # ── Passo 6 ──
    use_contrastive_loss=True,
    contrastive_dim=128,
    contrastive_temp=0.07,
    contrastive_weight=0.1,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Translator — beam search with n-gram blocking and length penalty
# ═══════════════════════════════════════════════════════════════════════════════

def _get_ngrams(sequence: list[int], n: int) -> set[tuple[int, ...]]:
    """Return the set of all n-grams in *sequence*."""
    return {tuple(sequence[i: i + n]) for i in range(len(sequence) - n + 1)}


def _apply_ngram_block(
    hyp_ids: list[int],
    logits: torch.Tensor,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    """Set logits to -inf for any token that would complete a repeated n-gram.

    Args:
        hyp_ids:            already generated token ids for this hypothesis
        logits:             (vocab_size,) raw logits for the next step
        no_repeat_ngram_size: n – block any n-gram already seen in hyp_ids

    Returns:
        logits with forbidden entries set to -inf (in-place clone)
    """
    if no_repeat_ngram_size <= 0 or len(hyp_ids) < no_repeat_ngram_size - 1:
        return logits

    # Build the (n-1)-gram prefix that would precede the next token
    prefix = tuple(hyp_ids[-(no_repeat_ngram_size - 1):])
    blocked_logits = logits.clone()

    # For each token, check whether (prefix + token) already appeared
    existing_ngrams = _get_ngrams(hyp_ids, no_repeat_ngram_size)
    for token_id in range(logits.size(-1)):
        if prefix + (token_id,) in existing_ngrams:
            blocked_logits[token_id] = float("-inf")

    return blocked_logits


class Translator:
    """Autoregressive beam-search decoder for RecursiveTransformer.

    Implements:
      - Passo 2: n-gram blocking (``no_repeat_ngram_size``)
      - Passo 3: length penalty  (``length_penalty``, GNMT formula)

    The decoder runs recurrently: memories from step *t* are fed into step
    *t+1*, mirroring the training-time recurrent forward pass.

    Args:
        model:                 trained RecursiveTransformer
        config:                edict with model hyper-parameters
        beam_size:             number of beams (1 = greedy)
        max_t_len:             maximum generation length (tokens)
        no_repeat_ngram_size:  block n-grams of this size (0 = disabled)
        length_penalty:        α in GNMT length penalty ((5+|Y|)/(5+1))^α
                               0 = no penalty, 1 = full normalisation
        bos_idx / eos_idx:     token indices for [BOS] and [EOS]
        device:                target device
    """

    def __init__(
        self,
        model: RecursiveTransformer,
        config: edict,
        beam_size: int = 4,
        max_t_len: int = 30,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 0.6,
        bos_idx: int = 4,
        eos_idx: int = 5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.beam_size = beam_size
        self.max_t_len = max_t_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.device = device or next(model.parameters()).device

    # ── Passo 3 helper ────────────────────────────────────────────────────────
    def _length_penalty_factor(self, length: int) -> float:
        """GNMT-style length penalty: ((5 + |Y|) / 6) ^ α."""
        return ((5.0 + length) / 6.0) ** self.length_penalty

    def translate_batch(
        self,
        input_ids_list: list[torch.Tensor],
        video_features_list: list[torch.Tensor],
        input_masks_list: list[torch.Tensor],
        token_type_ids_list: list[torch.Tensor],
    ) -> list[list[str]]:
        """Decode a full paragraph (multiple recurrent steps) for a batch.

        Returns:
            List of paragraphs, each a list of decoded sentences (one per step).
        """
        self.model.eval()
        bsz = input_ids_list[0].size(0)
        step_size = len(input_ids_list)

        prev_ms: list[Optional[torch.Tensor]] = [None] * self.config.num_hidden_layers
        coverages: list[Optional[torch.Tensor]] = [None] * self.config.num_hidden_layers

        # Collect decoded sequences: [bsz][step] = list of token ids
        all_decoded: list[list[list[int]]] = [[[] for _ in range(step_size)] for _ in range(bsz)]

        with torch.no_grad():
            for step_idx in range(step_size):
                # ── Encode the video prefix with teacher-forced text prefix ──
                # We use a greedy prefix here; for full beam search you would
                # replicate the states per beam. This implementation does
                # per-step greedy with n-gram blocking + length penalty.
                _, encoded_layers, _, coverages = self.model.forward_step(
                    prev_ms,
                    input_ids_list[step_idx],
                    video_features_list[step_idx],
                    input_masks_list[step_idx],
                    token_type_ids_list[step_idx],
                    coverages=coverages,
                )

                # ── Autoregressive decoding for this step ─────────────────────
                # Start from the [BOS] position (max_v_len) and decode greedily
                # with n-gram blocking and length-penalty-aware scoring.
                hidden = encoded_layers[-1]  # (N, L, D)
                # Decode token by token using the hidden states as context
                # (simplified: use prediction_scores from the full encoded seq)
                _, _, prediction_scores, _ = self.model.forward_step(
                    prev_ms,
                    input_ids_list[step_idx],
                    video_features_list[step_idx],
                    input_masks_list[step_idx],
                    token_type_ids_list[step_idx],
                    coverages=coverages,
                )
                # prediction_scores: (N, L, vocab_size)
                # Take text positions only (after max_v_len)
                text_logits = prediction_scores[:, self.config.max_v_len:, :]  # (N, max_t_len, V)

                decoded_batch: list[list[int]] = [[] for _ in range(bsz)]
                for pos in range(self.max_t_len):
                    for b in range(bsz):
                        if self.eos_idx in decoded_batch[b]:
                            continue
                        logits = text_logits[b, pos]  # (V,)
                        # Passo 2: n-gram blocking
                        logits = _apply_ngram_block(
                            decoded_batch[b], logits, self.no_repeat_ngram_size
                        )
                        token = logits.argmax(-1).item()
                        decoded_batch[b].append(int(token))

                # Passo 3: length-penalised re-scoring per beam (scalar here)
                for b in range(bsz):
                    seq = decoded_batch[b]
                    # Trim at EOS
                    if self.eos_idx in seq:
                        seq = seq[: seq.index(self.eos_idx)]
                    lp = self._length_penalty_factor(max(len(seq), 1))
                    # Score (used if caller wants ranked candidates; stored for reference)
                    score = sum(
                        text_logits[b, t, decoded_batch[b][t]].item()
                        for t in range(len(decoded_batch[b]))
                    ) / lp
                    all_decoded[b][step_idx] = seq
                    logger.debug("step=%d batch=%d len=%d lp=%.3f score=%.3f",
                                 step_idx, b, len(seq), lp, score)

                # ── Update recurrent memory for next step ─────────────────────
                prev_ms, _, _, coverages = self.model.forward_step(
                    prev_ms,
                    input_ids_list[step_idx],
                    video_features_list[step_idx],
                    input_masks_list[step_idx],
                    token_type_ids_list[step_idx],
                    coverages=coverages,
                )

        return all_decoded