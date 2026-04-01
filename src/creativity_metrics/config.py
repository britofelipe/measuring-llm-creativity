from dataclasses import dataclass, field
from typing import Optional, Sequence

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 64
    normalize_embeddings: bool = True

@dataclass
class MetricConfig:
    mattr_window: int = 50
    distinct_n: int = 2
    rarity_ngram_n: int = 2
    bertscore_lang: str = "fr"
    neighbor_k: int = 5
    min_sentence_count: int = 2
    min_tokens_for_metrics: int = 3

@dataclass
class PipelineConfig:
    dataset_path: str = "hf://datasets/ministere-culture/comparia-reactions/reactions.parquet"
    sample_size: Optional[int] = None
    random_state: int = 42
    text_columns: Sequence[str] = field(default_factory=lambda: [
        "question_content",
        "response_content",
        "system_prompt",
    ])
    required_columns: Sequence[str] = field(default_factory=lambda: [
        "id",
        "question_id",
        "msg_index",
        "question_content",
        "response_content",
        "system_prompt",
        "creative",
        "useful",
        "complete",
        "incorrect",
        "superficial",
        "instructions_not_followed",
        "model_a_name",
        "model_b_name",
        "refers_to_model",
    ])

@dataclass
class MetricConfig:
    mattr_window: int = 50
    distinct_n: int = 2
    rarity_ngram_n: int = 2
    bertscore_lang: str = "fr"
    neighbor_k: int = 5
    min_sentence_count: int = 2
    min_tokens_for_metrics: int = 3
    rarity_reference_path: Optional[str] = None