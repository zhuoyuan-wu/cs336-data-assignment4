# scripts/tokenize_dataset.py
from __future__ import annotations

import argparse
from pathlib import Path
import multiprocessing
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def tokenize_doc_worker(doc_text: str):
    # 为多进程提供 worker
    global _TOKENIZER
    if doc_text.strip():
        return _TOKENIZER.encode(doc_text) + [_TOKENIZER.eos_token_id]
    return []


def init_worker():
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to clean text file")
    parser.add_argument("--output-bin", type=str, required=True, help="Path to output .bin file")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw text from {input_path}...")
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        # 按双换行切分成独立文档
        content = f.read()
        docs = [d.strip() for d in content.split("\n\n") if d.strip()]

    print(f"Tokenizing {len(docs)} documents using {args.workers} workers...")
    with multiprocessing.Pool(processes=args.workers, initializer=init_worker) as pool:
        results = list(
            tqdm(
                pool.imap(tokenize_doc_worker, docs, chunksize=128),
                total=len(docs),
                desc="Tokenizing"
            )
        )

    all_ids = [tok_id for sublist in results for tok_id in sublist]
    print(f"\nTotal tokens produced: {len(all_ids):,}")

    # GPT-2 词表大小 50257，完全适应 np.uint16 (0 ~ 65535)
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_path)
    print(f"Successfully serialized to {output_path} (Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()