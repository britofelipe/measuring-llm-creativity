from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .text_utils import split_sentences, tokenize, lcs_length
from .embeddings import cosine_similarity


def rouge_l_recall(prompt: str, response: str) -> float:
    a = tokenize(prompt)
    b = tokenize(response)
    if not a or not b:
        return np.nan
    lcs = lcs_length(a, b)
    return lcs / max(len(a), 1)


def bertscore_f1_batch(prompts: List[str], responses: List[str], lang: str = "fr") -> np.ndarray:
    if len(prompts) != len(responses):
        raise ValueError("prompts et responses doivent avoir la meme longueur.")

    prompts_clean = ["" if p is None else str(p).strip() for p in prompts]
    responses_clean = ["" if r is None else str(r).strip() for r in responses]

    valid_idx = [
        i for i, (p, r) in enumerate(zip(prompts_clean, responses_clean))
        if p and r
    ]

    # Evite les crashs de bert-score sur chaines vides.
    if not valid_idx:
        return np.full(len(prompts), np.nan, dtype=float)

    try:
        from bert_score import score as bert_score
    except ImportError as e:
        raise ImportError(
            "bert-score est requis pour BERTScore. Installez-le avec `pip install bert-score`."
        ) from e

    valid_prompts = [prompts_clean[i] for i in valid_idx]
    valid_responses = [responses_clean[i] for i in valid_idx]

    _, _, f1 = bert_score(
        cands=valid_responses,
        refs=valid_prompts,
        lang=lang,
        verbose=False,
        rescale_with_baseline=True,
        use_fast_tokenizer=False,
    )

    scores = np.full(len(prompts), np.nan, dtype=float)
    scores[np.array(valid_idx)] = f1.detach().cpu().numpy()
    return scores


def local_coherence_from_sentence_embeddings(sentence_vectors: np.ndarray) -> float:
    if sentence_vectors.shape[0] < 2:
        return np.nan
    sims = []
    for i in range(sentence_vectors.shape[0] - 1):
        sim = float(cosine_similarity(sentence_vectors[i], sentence_vectors[i + 1])[0, 0])
        sims.append(sim)
    return float(np.mean(sims)) if sims else np.nan


def add_value_metrics(
    df: pd.DataFrame,
    sentence_embedder,
    bertscore_lang: str = "fr",
) -> pd.DataFrame:
    out = df.copy()

    out["value_rouge_l_prompt_response"] = [
        rouge_l_recall(q, r) for q, r in zip(out["question_content"], out["response_content"])
    ]

    out["value_bertscore_f1"] = bertscore_f1_batch(
        prompts=out["question_content"].tolist(),
        responses=out["response_content"].tolist(),
        lang=bertscore_lang,
    )

    coherence_scores = []
    for response in out["response_content"].tolist():
        sents = split_sentences(response)
        if len(sents) < 2:
            coherence_scores.append(np.nan)
            continue
        sent_vecs = sentence_embedder.encode(sents)
        coherence_scores.append(local_coherence_from_sentence_embeddings(sent_vecs))
    out["value_local_coherence"] = coherence_scores

    return out
