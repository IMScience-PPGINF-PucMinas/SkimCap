"""
build_vocab_clip.py — gera o anet_vocab_clip.pt compatível com VLTinT.

Coleta todos os tokens únicos presentes nos arquivos lang_feature/*.json e
encoda cada um com o text encoder do CLIP (ViT-B/16), produzindo um dict
{word: np.ndarray(D_clip,)} salvo via torch.save.

Uso:
    python build_vocab_clip.py \
        --lang_feature_dir data/anet/clip_b16/lang_feature \
        --output_path      cache/anet_vocab_clip.pt \
        --clip_model       ViT-B/16 \
        --batch_size       256

Dependências:
    pip install git+https://github.com/openai/CLIP.git
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import tqdm


def collect_all_tokens(lang_feature_dir: str) -> list[str]:
    """Varre todos os JSON de lang features e retorna tokens únicos ordenados."""
    token_set: set[str] = set()
    files = glob(os.path.join(lang_feature_dir, "*.json"))
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo .json encontrado em: {lang_feature_dir}"
        )
    for path in tqdm.tqdm(files, desc="Coletando tokens"):
        with open(path) as f:
            data = json.load(f)
        # data: list[list[str]] — (N_clips, K_tokens)
        for clip_tokens in data:
            token_set.update(clip_tokens)
    tokens = sorted(token_set)
    print(f"Tokens únicos encontrados: {len(tokens)}")
    return tokens


def encode_tokens_clip(
    tokens: list[str],
    clip_model_name: str,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Encoda cada token com o text encoder do CLIP e retorna {token: vetor}."""
    try:
        import clip
    except ImportError:
        raise ImportError(
            "CLIP não instalado. Execute:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    model, _ = clip.load(clip_model_name, device=device)
    model.eval()

    vocab: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for i in tqdm.tqdm(
            range(0, len(tokens), batch_size), desc="Encodando com CLIP"
        ):
            batch_tokens = tokens[i : i + batch_size]
            # clip.tokenize trunca automaticamente para 77 tokens
            text_input = clip.tokenize(batch_tokens, truncate=True).to(device)
            features = model.encode_text(text_input)           # (B, D)
            features = features / features.norm(dim=-1, keepdim=True)  # L2-norm
            features = features.cpu().float().numpy()

            for token, vec in zip(batch_tokens, features):
                vocab[token] = vec.astype(np.float32)

    return vocab


def main():
    parser = argparse.ArgumentParser(
        description="Gera anet_vocab_clip.pt a partir dos lang_feature JSONs"
    )
    parser.add_argument(
        "--lang_feature_dir",
        required=True,
        help="Diretório com os arquivos <video_name>.json de lang features",
    )
    parser.add_argument(
        "--output_path",
        default="cache/anet_vocab_clip.pt",
        help="Caminho de saída para o vocab_clip.pt",
    )
    parser.add_argument(
        "--clip_model",
        default="ViT-B/16",
        choices=["ViT-B/16", "ViT-B/32", "ViT-L/14"],
        help="Versão do modelo CLIP a usar",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size para o text encoder do CLIP",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda / cpu)",
    )
    opt = parser.parse_args()

    os.makedirs(os.path.dirname(opt.output_path) or ".", exist_ok=True)

    print(f"Coletando tokens de: {opt.lang_feature_dir}")
    tokens = collect_all_tokens(opt.lang_feature_dir)

    print(f"Encodando {len(tokens)} tokens com CLIP {opt.clip_model} em {opt.device}…")
    vocab = encode_tokens_clip(tokens, opt.clip_model, opt.batch_size, opt.device)

    torch.save(vocab, opt.output_path)
    D = next(iter(vocab.values())).shape[0]
    print(f"Salvo em: {opt.output_path}  ({len(vocab)} tokens, D={D})")
    print("Use com:  --vocab_clip_path", opt.output_path, "--lang_feature_size", D)


if __name__ == "__main__":
    main()