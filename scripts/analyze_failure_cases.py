from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze false positives / false negatives using saved logistic "
            "coefficients (no re-training)."
        )
    )
    parser.add_argument(
        "--scored-parquet",
        type=str,
        default="outputs_logreg2/comparia_reactions_with_metrics.parquet",
        help="Parquet file with metrics and labels.",
    )
    parser.add_argument(
        "--coefficients-csv",
        type=str,
        default="outputs_logreg2/logreg_coefficients.csv",
        help="CSV with columns: feature, coefficient.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="outputs_logreg2/logreg_train_test_metrics.csv",
        help="CSV that contains intercept and threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override threshold.",
    )
    parser.add_argument(
        "--intercept",
        type=float,
        default=None,
        help="Optional override intercept.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="How many FP and FN examples to print.",
    )
    parser.add_argument(
        "--top-contrib",
        type=int,
        default=5,
        help="How many top feature contributions to print per case.",
    )
    parser.add_argument(
        "--max-question-chars",
        type=int,
        default=500,
        help="Maximum chars to print for question content.",
    )
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=1600,
        help="Maximum chars to print for response content.",
    )
    parser.add_argument(
        "--max-turn-chars",
        type=int,
        default=280,
        help="Maximum chars per conversation turn in excerpt.",
    )
    parser.add_argument(
        "--conversation-window",
        type=int,
        default=2,
        help="Turns before/after msg_index to show.",
    )
    parser.add_argument(
        "--include-system-prompt",
        action="store_true",
        help="Include system_prompt in printed examples.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Optional path to save the report text.",
    )
    return parser.parse_args()


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def clip_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def load_coefficients(path: str) -> Dict[str, float]:
    coef_df = pd.read_csv(path)
    required = {"feature", "coefficient"}
    if not required.issubset(coef_df.columns):
        raise ValueError(f"Missing required columns in coefficients CSV: {required}")
    return {
        str(row["feature"]): float(row["coefficient"])
        for _, row in coef_df.iterrows()
    }


def load_threshold_and_intercept(
    metrics_csv: str,
    threshold_override: float | None,
    intercept_override: float | None,
) -> Tuple[float, float, str]:
    threshold_mode = "unknown"
    threshold = threshold_override
    intercept = intercept_override

    if threshold is not None and intercept is not None:
        return float(threshold), float(intercept), threshold_mode

    metrics_df = pd.read_csv(metrics_csv)
    if metrics_df.empty:
        raise ValueError("metrics CSV is empty.")

    if "split" in metrics_df.columns and "train" in metrics_df["split"].values:
        row = metrics_df.loc[metrics_df["split"] == "train"].iloc[0]
    else:
        row = metrics_df.iloc[0]

    if threshold is None:
        if "threshold" not in metrics_df.columns:
            raise ValueError("threshold not found in metrics CSV.")
        threshold = float(row["threshold"])

    if intercept is None:
        if "intercept" not in metrics_df.columns:
            raise ValueError("intercept not found in metrics CSV.")
        intercept = float(row["intercept"])

    if "threshold_mode" in metrics_df.columns:
        threshold_mode = str(row["threshold_mode"])

    return float(threshold), float(intercept), threshold_mode


def ensure_feature_columns(df: pd.DataFrame, feature_names: Iterable[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    missing = []
    for feature in feature_names:
        if feature in out.columns:
            continue
        raw_col = feature[:-5] if feature.endswith("_norm") else None
        if raw_col and raw_col in out.columns:
            # Lazy fallback: robust normalization exactly as in the training pipeline.
            from creativity_metrics.scoring import normalize_metric

            out[feature] = normalize_metric(out[raw_col])
        else:
            missing.append(feature)
    return out, missing


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def infer_conversation_column(row: pd.Series) -> str | None:
    refers_to_model = safe_text(row.get("refers_to_model"))
    model_a = safe_text(row.get("model_a_name"))
    model_b = safe_text(row.get("model_b_name"))

    if refers_to_model and refers_to_model == model_a and "conversation_a" in row.index:
        return "conversation_a"
    if refers_to_model and refers_to_model == model_b and "conversation_b" in row.index:
        return "conversation_b"

    # Fallback based on model_pos when present.
    model_pos = safe_text(row.get("model_pos")).upper()
    if model_pos == "A" and "conversation_a" in row.index:
        return "conversation_a"
    if model_pos == "B" and "conversation_b" in row.index:
        return "conversation_b"

    return None


def to_turn_list(value: object) -> List[dict]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"role": "unknown", "content": safe_text(item)})
        return out
    return []


def render_conversation_excerpt(
    row: pd.Series,
    max_turn_chars: int,
    conversation_window: int,
) -> str:
    conv_col = infer_conversation_column(row)
    if not conv_col:
        return "(conversation unavailable)"

    turns = to_turn_list(row.get(conv_col))
    if not turns:
        return "(conversation unavailable)"

    msg_index_raw = row.get("msg_index")
    try:
        msg_index = int(msg_index_raw)
    except Exception:
        msg_index = None

    if msg_index is None:
        start, end = 0, min(len(turns), 2 * conversation_window + 1)
    else:
        start = max(0, msg_index - conversation_window)
        end = min(len(turns), msg_index + conversation_window + 1)

    lines = [f"[source: {conv_col}, turns {start}..{max(start, end-1)}]"]
    for i in range(start, end):
        turn = turns[i]
        role = safe_text(turn.get("role")) or "unknown"
        content = clip_text(safe_text(turn.get("content")), max_turn_chars)
        marker = ""
        if msg_index is not None and i == msg_index:
            marker = "  <== msg_index"
        lines.append(f"  - [{i}] {role}: {content}{marker}")
    return "\n".join(lines)


def top_contributions(
    row: pd.Series,
    coefficients: Dict[str, float],
    top_n: int,
) -> List[Tuple[str, float, float, float]]:
    contrib = []
    for feature, coef in coefficients.items():
        value = row.get(feature)
        if pd.isna(value):
            continue
        value_f = float(value)
        c = coef * value_f
        contrib.append((feature, coef, value_f, c))
    contrib.sort(key=lambda x: abs(x[3]), reverse=True)
    return contrib[:top_n]


def case_lines(
    row: pd.Series,
    case_label: str,
    threshold: float,
    coefficients: Dict[str, float],
    top_contrib_n: int,
    max_question_chars: int,
    max_response_chars: int,
    max_turn_chars: int,
    conversation_window: int,
    include_system_prompt: bool,
) -> List[str]:
    lines: List[str] = []
    row_id = safe_text(row.get("id"))
    question_id = safe_text(row.get("question_id"))
    msg_index = safe_text(row.get("msg_index"))
    model_name = safe_text(row.get("refers_to_model"))
    prob = float(row["pred_prob_from_csv"])
    pred = bool(row["pred_from_csv"])
    creative = bool(row["creative_bool"])

    lines.append(f"### {case_label} | id={row_id} | question_id={question_id} | msg_index={msg_index}")
    lines.append(
        f"creative={creative} | pred={pred} | prob={prob:.4f} | threshold={threshold:.4f} | model={model_name}"
    )
    lines.append("")
    lines.append("Question:")
    lines.append(clip_text(safe_text(row.get("question_content")), max_question_chars))
    lines.append("")
    lines.append("Response:")
    lines.append(clip_text(safe_text(row.get("response_content")), max_response_chars))
    lines.append("")

    if include_system_prompt:
        lines.append("System prompt:")
        lines.append(clip_text(safe_text(row.get("system_prompt")), max_question_chars))
        lines.append("")

    lines.append("Conversation excerpt:")
    lines.append(
        render_conversation_excerpt(
            row=row,
            max_turn_chars=max_turn_chars,
            conversation_window=conversation_window,
        )
    )
    lines.append("")

    lines.append("Top feature contributions (coef * feature):")
    for feature, coef, value, contrib in top_contributions(row, coefficients, top_contrib_n):
        lines.append(
            f"- {feature}: coef={coef:+.4f}, value={value:.4f}, contrib={contrib:+.4f}"
        )
    lines.append("")
    return lines


def build_report(args: argparse.Namespace) -> str:
    df = pd.read_parquet(args.scored_parquet)
    if "creative" not in df.columns:
        raise ValueError("Column 'creative' not found in parquet.")

    coefficients = load_coefficients(args.coefficients_csv)
    threshold, intercept, threshold_mode = load_threshold_and_intercept(
        metrics_csv=args.metrics_csv,
        threshold_override=args.threshold,
        intercept_override=args.intercept,
    )

    df, missing_features = ensure_feature_columns(df, coefficients.keys())
    if missing_features:
        raise ValueError(
            "Missing features required by coefficients: "
            + ", ".join(sorted(missing_features))
        )

    feature_cols = list(coefficients.keys())
    eligible_mask = df[feature_cols].notna().all(axis=1)
    work = df.loc[eligible_mask].copy()

    if work.empty:
        raise ValueError("No eligible rows after requiring complete feature columns.")

    coef_series = pd.Series(coefficients, dtype=float)
    logit = work[feature_cols].mul(coef_series, axis=1).sum(axis=1) + intercept
    work["pred_prob_from_csv"] = sigmoid(logit.to_numpy())
    work["pred_from_csv"] = work["pred_prob_from_csv"] >= threshold
    work["creative_bool"] = work["creative"].astype(bool)

    fp = work[(work["pred_from_csv"]) & (~work["creative_bool"])].copy()
    fn = work[(~work["pred_from_csv"]) & (work["creative_bool"])].copy()

    fp = fp.sort_values("pred_prob_from_csv", ascending=False)
    fn = fn.sort_values("pred_prob_from_csv", ascending=True)

    lines: List[str] = []
    lines.append("# Failure Cases Analysis (No Re-Training)")
    lines.append("")
    lines.append(f"scored_parquet: {args.scored_parquet}")
    lines.append(f"coefficients_csv: {args.coefficients_csv}")
    lines.append(f"metrics_csv: {args.metrics_csv}")
    lines.append(
        f"threshold={threshold:.4f} (mode={threshold_mode}), intercept={intercept:.6f}"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- total rows in parquet: {len(df)}")
    lines.append(f"- eligible rows (all required features present): {len(work)}")
    lines.append(
        f"- positive rate (creative=True) on eligible rows: {work['creative_bool'].mean():.4f}"
    )
    lines.append(f"- false positives: {len(fp)}")
    lines.append(f"- false negatives: {len(fn)}")
    lines.append("")

    if "creativity_index_logreg" in work.columns:
        delta = (
            work["pred_prob_from_csv"] - pd.to_numeric(work["creativity_index_logreg"], errors="coerce")
        ).abs()
        lines.append("## Consistency Check")
        lines.append(
            f"- max |recomputed_prob - creativity_index_logreg|: {float(np.nanmax(delta.values)):.8f}"
        )
        lines.append(
            f"- mean |recomputed_prob - creativity_index_logreg|: {float(np.nanmean(delta.values)):.8f}"
        )
        lines.append("")

    lines.append(f"## Top {args.top_k} False Positives")
    if fp.empty:
        lines.append("- none")
    else:
        for i, (_, row) in enumerate(fp.head(args.top_k).iterrows(), start=1):
            lines.extend(
                case_lines(
                    row=row,
                    case_label=f"FP #{i}",
                    threshold=threshold,
                    coefficients=coefficients,
                    top_contrib_n=args.top_contrib,
                    max_question_chars=args.max_question_chars,
                    max_response_chars=args.max_response_chars,
                    max_turn_chars=args.max_turn_chars,
                    conversation_window=args.conversation_window,
                    include_system_prompt=args.include_system_prompt,
                )
            )

    lines.append(f"## Top {args.top_k} False Negatives")
    if fn.empty:
        lines.append("- none")
    else:
        for i, (_, row) in enumerate(fn.head(args.top_k).iterrows(), start=1):
            lines.extend(
                case_lines(
                    row=row,
                    case_label=f"FN #{i}",
                    threshold=threshold,
                    coefficients=coefficients,
                    top_contrib_n=args.top_contrib,
                    max_question_chars=args.max_question_chars,
                    max_response_chars=args.max_response_chars,
                    max_turn_chars=args.max_turn_chars,
                    conversation_window=args.conversation_window,
                    include_system_prompt=args.include_system_prompt,
                )
            )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args)
    print(report)

    if args.output_report:
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
