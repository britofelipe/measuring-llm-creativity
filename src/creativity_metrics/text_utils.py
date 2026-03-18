from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Tuple


_TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    return _TOKEN_RE.findall(text)


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if len(sents) <= 1:
        # fallback naïf
        sents = [seg.strip() for seg in re.split(r"[\n;]+", text) if seg.strip()]
    return sents if sents else ([text] if text else [])


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n or n <= 0:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def count_ngrams(tokens_list: Iterable[List[str]], n: int) -> Counter:
    counter = Counter()
    for tokens in tokens_list:
        counter.update(get_ngrams(tokens, n))
    return counter


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]
