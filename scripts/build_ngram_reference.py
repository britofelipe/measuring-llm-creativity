from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from collections import Counter

from datasets import load_dataset

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from creativity_metrics.text_utils import tokenize, get_ngrams


def build_reference_counts_from_hf(
    dataset_name: str,
    split: str,
    text_col: str,
    n: int,
    limit: int | None,
    streaming: bool,
) -> Counter:
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    counter = Counter()

    for i, row in enumerate(ds):
        text = row.get(text_col, "") or ""
        tokens = tokenize(text)
        counter.update(get_ngrams(tokens, n=n))

        if limit is not None and (i + 1) >= limit:
            break

    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Construire une référence de n-grammes depuis un dataset HF")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--ngram-n", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    counts = build_reference_counts_from_hf(
        dataset_name=args.dataset_name,
        split=args.split,
        text_col=args.text_col,
        n=args.ngram_n,
        limit=args.limit,
        streaming=args.streaming,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(counts, f)

    print(f"Référence sauvegardée dans: {output_path}")
    print(f"Nombre de n-grammes distincts: {len(counts)}")
    print(f"Nombre total de n-grammes: {sum(counts.values())}")


if __name__ == "__main__":
    main()