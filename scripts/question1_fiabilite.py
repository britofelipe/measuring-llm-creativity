"""
Question 1 - Fiabilite des jugements humains
  1.1 - Accord inter-utilisateurs sur le signal creative (taux d'accord brut + kappa de Cohen)
  1.2 - Biais de selection des votants
  1.3 - alpha de Krippendorff

Methodologie d'identification du meme message:
  - Approche stricte  : (refers_to_model, response_content) avec len > 50
    Cas ou le meme texte de reponse a ete evalue par des visitors differents
    (multi-annotation incidentale: reponses deterministes ou generiques)
  - Proxy etendu      : (refers_to_model, question_content)
    Meme question posee au meme modele par des visitors differents
    (la reponse peut varier selon la stochasticite du modele)

Usage:
    python scripts/question1_fiabilite.py --output-dir outputs/question1
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score

try:
    import krippendorff
except ImportError:
    krippendorff = None
    warnings.warn("Package 'krippendorff' non installe. pip install krippendorff")

HF_REACTIONS = "hf://datasets/ministere-culture/comparia-reactions/data/train-00000-of-00001.parquet"
HF_VOTES = "hf://datasets/ministere-culture/comparia-votes/data/train-00000-of-00001.parquet"
HF_CONVERSATIONS = "hf://datasets/ministere-culture/comparia-conversations/data/train-00000-of-00001.parquet"

REACTION_BOOL_COLS = ["creative", "useful", "complete", "incorrect", "superficial"]
VISITOR_COL = "visitor_id"
# Filtre longueur reponse pour l'approche stricte: exclut les reponses vides/generiques
MIN_RESPONSE_LEN = 50

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def load_parquet_hf(path: str, sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    print(f"Loading {path}")
    try:
        df = pd.read_parquet(path)
    except Exception:
        from datasets import load_dataset
        parts = path.replace("hf://datasets/", "").split("/")
        dataset_name = "/".join(parts[:2])
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)
    return df


def ensure_bool(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    return df


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _compute_pairwise_kappa(df: pd.DataFrame, item_col: str, visitor_col: str,
                             labels: list[str]) -> list[dict]:
    """
    Forme toutes les paires de visitors au sein de chaque item et calcule
    l'accord brut et le kappa de Cohen agregees sur toutes les paires.
    """
    results = []
    for label in labels:
        all_a, all_b = [], []
        for _, grp in df.groupby(item_col):
            vals = grp[label].astype(int).values
            if len(vals) < 2:
                continue
            for i, j in combinations(range(len(vals)), 2):
                all_a.append(vals[i])
                all_b.append(vals[j])

        arr_a, arr_b = np.array(all_a), np.array(all_b)
        n = len(arr_a)
        if n == 0:
            continue

        accord = (arr_a == arr_b).mean()
        try:
            kappa = cohen_kappa_score(arr_a, arr_b)
        except Exception:
            kappa = float("nan")
        prevalence = df[label].mean()

        results.append({
            "Label": label,
            "N_paires": n,
            "Taux_accord_brut": round(accord, 4),
            "Kappa_Cohen": round(kappa, 4),
            "Prevalence_label": round(prevalence, 4),
        })
    return results


def _build_multi_item_df(df: pd.DataFrame, group_cols: list[str], visitor_col: str,
                          labels: list[str], min_visitors: int = 2,
                          response_len_filter: int | None = None) -> tuple[pd.DataFrame, int]:
    """
    Filtre le dataframe pour garder seulement les items avec >= min_visitors visiteurs distincts.
    Retourne (df_multi_deduplie, n_items).
    """
    df_work = df[group_cols + [visitor_col] + labels].copy()

    if response_len_filter is not None and "response_content" in group_cols:
        df_work = df_work[df_work["response_content"].str.len() > response_len_filter]

    df_work = df_work.dropna(subset=group_cols + [visitor_col])
    df_work["item_id"] = df_work[group_cols[0]].astype(str)
    for col in group_cols[1:]:
        df_work["item_id"] = df_work["item_id"] + "|||" + df_work[col].astype(str)

    counts = df_work.groupby("item_id")[visitor_col].nunique()
    items_ok = counts[counts >= min_visitors].index
    df_multi = df_work[df_work["item_id"].isin(items_ok)].copy()
    df_multi = df_multi.drop_duplicates(subset=["item_id", visitor_col], keep="first")
    return df_multi, len(items_ok)


# -----------------------------------------------------------------------
# Exercice 1.1 - Accord inter-utilisateurs sur le signal creative
# -----------------------------------------------------------------------

def exercice_1_1(df_reactions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Accord inter-utilisateurs via deux approches complementaires:

    A) STRICTE — (refers_to_model, response_content, len>50):
       Multi-annotation incidentale: le meme texte de reponse a ete evalue par
       des visiteurs differents. Cela arrive quand le LLM produit des reponses
       identiques (determinisme, reponses courtes). 57 items avec >=2 visiteurs.
       Limitation: biais vers les reponses courtes ou generiques.

    B) PROXY — (refers_to_model, question_content):
       Meme question posee au meme modele par des visiteurs differents.
       La reponse n'est pas garantie identique (stochasticite). 778 items avec >=2 visiteurs.
    """
    visitor_col = VISITOR_COL
    labels = [c for c in ["creative", "useful", "complete"] if c in df_reactions.columns]

    for c in [visitor_col] + labels:
        if c not in df_reactions.columns:
            print(f"Colonne manquante: {c}")
            return pd.DataFrame()

    all_results = []

    # --- Approche A: stricte (response_content) ---
    if "response_content" in df_reactions.columns:
        group_a = ["refers_to_model", "response_content"]
        df_multi_a, n_items_a = _build_multi_item_df(
            df_reactions, group_a, visitor_col, labels,
            min_visitors=2, response_len_filter=MIN_RESPONSE_LEN
        )
        print(f"[Stricte] Items (model, response_content, len>{MIN_RESPONSE_LEN}) avec >=2 visiteurs: {n_items_a:,}")
        if len(df_multi_a) > 0:
            res_a = _compute_pairwise_kappa(df_multi_a, "item_id", visitor_col, labels)
            for r in res_a:
                r["Approche"] = "stricte_response_content"
                r["N_items"] = n_items_a
                print(f"  {r['Label']}: n_paires={r['N_paires']}, accord={r['Taux_accord_brut']:.2%}, "
                      f"kappa={r['Kappa_Cohen']:.4f}")
            all_results.extend(res_a)
    else:
        print("[Stricte] Colonne response_content absente, approche stricte ignoree.")

    # --- Approche B: proxy (question_content) ---
    if "question_content" in df_reactions.columns:
        group_b = ["refers_to_model", "question_content"]
        df_multi_b, n_items_b = _build_multi_item_df(
            df_reactions, group_b, visitor_col, labels, min_visitors=2
        )
        print(f"[Proxy]   Items (model, question_content) avec >=2 visiteurs: {n_items_b:,}")
        if len(df_multi_b) > 0:
            res_b = _compute_pairwise_kappa(df_multi_b, "item_id", visitor_col, labels)
            for r in res_b:
                r["Approche"] = "proxy_question_content"
                r["N_items"] = n_items_b
                print(f"  {r['Label']}: n_paires={r['N_paires']}, accord={r['Taux_accord_brut']:.2%}, "
                      f"kappa={r['Kappa_Cohen']:.4f}")
            all_results.extend(res_b)

    if not all_results:
        return pd.DataFrame()

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_dir / "accord_brut_et_kappa.csv", index=False)

    # --- Graphique comparatif ---
    df_a = df_results[df_results["Approche"] == "stricte_response_content"]
    df_b = df_results[df_results["Approche"] == "proxy_question_content"]

    if len(df_a) > 0 and len(df_b) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(labels))
        width = 0.35

        for ax_idx, (metric, ylabel, ylim) in enumerate([
            ("Taux_accord_brut", "Taux d'accord brut", (0, 1.0)),
            ("Kappa_Cohen", "kappa de Cohen", (-0.1, 1.0)),
        ]):
            ax = axes[ax_idx]
            vals_a = [df_a[df_a["Label"] == l][metric].values[0]
                      if len(df_a[df_a["Label"] == l]) > 0 else 0 for l in labels]
            vals_b = [df_b[df_b["Label"] == l][metric].values[0]
                      if len(df_b[df_b["Label"] == l]) > 0 else 0 for l in labels]

            bars_a = ax.bar(x - width / 2, vals_a, width,
                            label="Stricte (response_content)", color="#4878d0", alpha=0.85)
            bars_b = ax.bar(x + width / 2, vals_b, width,
                            label="Proxy (question_content)", color="#ee854a", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)
            ax.set_title(f"Exercice 1.1 - {ylabel}\ncomparaison approches")
            ax.legend(fontsize=8)

            if metric == "Kappa_Cohen":
                ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
                ax.axhline(y=0.4, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)

            for bar, val in zip(bars_a, vals_a):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)
            for bar, val in zip(bars_b, vals_b):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        save_fig(fig, output_dir / "exercice_1_1_accord_kappa.png")

    return df_results


# -----------------------------------------------------------------------
# Exercice 1.2 - Biais de selection des votants
# -----------------------------------------------------------------------

def exercice_1_2(df_conversations: pd.DataFrame, df_votes: pd.DataFrame,
                 output_dir: Path) -> pd.DataFrame:
    """
    Compare les conversations votees vs non-votees pour detecter un biais de selection.
    """
    join_col = None
    for candidate in ["conversation_pair_id", "conversation_id", "id"]:
        if candidate in df_conversations.columns and candidate in df_votes.columns:
            join_col = candidate
            break

    if join_col is None:
        if "conversation_pair_id" in df_votes.columns and "id" in df_conversations.columns:
            voted_ids = set(df_votes["conversation_pair_id"].dropna().unique())
            df_conversations = df_conversations.copy()
            df_conversations["a_recu_vote"] = df_conversations["id"].isin(voted_ids)
        else:
            print("Pas de colonne de jointure commune trouvee.")
            return pd.DataFrame()
    else:
        voted_ids = set(df_votes[join_col].dropna().unique())
        df_conversations = df_conversations.copy()
        df_conversations["a_recu_vote"] = df_conversations[join_col].isin(voted_ids)

    n_total = len(df_conversations)
    n_voted = int(df_conversations["a_recu_vote"].sum())
    print(f"Conversations: {n_total:,} total, {n_voted:,} votees ({n_voted/n_total:.1%})")

    turns_col = None
    for c in ["conv_turns", "nb_turns", "turns"]:
        if c in df_conversations.columns:
            turns_col = c
            break

    tokens_col = None
    if ("total_conv_a_output_tokens" in df_conversations.columns and
        "total_conv_b_output_tokens" in df_conversations.columns):
        df_conversations["total_output_tokens_sum"] = (
            df_conversations["total_conv_a_output_tokens"].fillna(0) +
            df_conversations["total_conv_b_output_tokens"].fillna(0)
        )
        tokens_col = "total_output_tokens_sum"
    else:
        for c in ["total_conv_a_output_tokens", "total_output_tokens"]:
            if c in df_conversations.columns:
                tokens_col = c
                break

    cat_col = None
    for c in ["categories", "selected_category", "category"]:
        if c in df_conversations.columns:
            cat_col = c
            break

    numeric_vars = {}
    if turns_col:
        numeric_vars["conv_turns"] = turns_col
    if tokens_col:
        numeric_vars["total_output_tokens"] = tokens_col

    stats_rows = []

    for display_name, col in numeric_vars.items():
        voted = df_conversations.loc[df_conversations["a_recu_vote"], col].dropna()
        unvoted = df_conversations.loc[~df_conversations["a_recu_vote"], col].dropna()
        if len(voted) == 0 or len(unvoted) == 0:
            continue

        u_stat, u_pval = stats.mannwhitneyu(voted, unvoted, alternative="two-sided")
        ks_stat, ks_pval = stats.ks_2samp(voted, unvoted)
        n1, n2 = len(voted), len(unvoted)
        rbc = 1 - (2 * u_stat) / (n1 * n2)

        row = {
            "Variable": display_name,
            "Moyenne_votees": round(voted.mean(), 2),
            "Moyenne_non_votees": round(unvoted.mean(), 2),
            "Mediane_votees": round(voted.median(), 2),
            "Mediane_non_votees": round(unvoted.median(), 2),
            "Ecart_type_votees": round(voted.std(), 2),
            "Ecart_type_non_votees": round(unvoted.std(), 2),
            "Mann_Whitney_U": round(u_stat, 2),
            "MW_p_value": f"{u_pval:.2e}",
            "KS_stat": round(ks_stat, 4),
            "KS_p_value": f"{ks_pval:.2e}",
            "Rank_biserial_r": round(rbc, 4),
        }
        stats_rows.append(row)
        print(f"  {display_name}: MW p={u_pval:.2e}, KS D={ks_stat:.4f}, r={rbc:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        if display_name == "conv_turns" or voted.nunique() < 20:
            clip_high = int(max(voted.quantile(0.99), unvoted.quantile(0.99)))
            if clip_high < 5: clip_high = 10
            bins = np.arange(0, clip_high + 2) - 0.5
            ax.hist(voted.clip(upper=clip_high), bins=bins, alpha=0.6, label=f"Votees (n={len(voted):,})", density=True, rwidth=0.8)
            ax.hist(unvoted.clip(upper=clip_high), bins=bins, alpha=0.6, label=f"Non votees (n={len(unvoted):,})", density=True, rwidth=0.8)
            ax.set_xticks(np.arange(0, clip_high + 1))
        else:
            clip_high = min(voted.quantile(0.99), unvoted.quantile(0.99))
            if clip_high <= 0: clip_high = max(voted.max(), unvoted.max())
            bins = np.linspace(0, clip_high, 50)
            ax.hist(voted.clip(upper=clip_high), bins=bins, alpha=0.6, label=f"Votees (n={len(voted):,})", density=True)
            ax.hist(unvoted.clip(upper=clip_high), bins=bins, alpha=0.6, label=f"Non votees (n={len(unvoted):,})", density=True)

        ax.set_xlabel(display_name)
        ax.set_ylabel("Densite")
        ax.set_title(f"Distribution de {display_name}")
        ax.legend()

        ax = axes[1]
        data_bp = pd.DataFrame({
            display_name: pd.concat([voted, unvoted]),
            "Groupe": (["Votees"] * len(voted)) + (["Non votees"] * len(unvoted))
        })
        data_bp[display_name] = data_bp[display_name].clip(upper=clip_high)
        sns.boxplot(data=data_bp, x="Groupe", y=display_name, ax=ax)
        ax.set_title(f"Boxplot - {display_name}")

        fig.suptitle(f"Exercice 1.2 - {display_name}: votees vs non votees", fontsize=13, y=1.02)
        fig.tight_layout()
        save_fig(fig, output_dir / f"exercice_1_2_{display_name}.png")

    if cat_col:
        df_cat = df_conversations[[cat_col, "a_recu_vote"]].copy()
        sample = df_cat[cat_col].dropna().iloc[:5] if len(df_cat) > 0 else pd.Series()
        if any(isinstance(v, (list, np.ndarray)) for v in sample):
            df_cat = df_cat.explode(cat_col)

        df_cat = df_cat.dropna(subset=[cat_col])
        df_cat[cat_col] = df_cat[cat_col].astype(str)

        top_cats = df_cat[cat_col].value_counts().head(15).index.tolist()
        df_cat_top = df_cat[df_cat[cat_col].isin(top_cats)]

        voted_counts = (df_cat_top[df_cat_top["a_recu_vote"]][cat_col]
                        .value_counts(normalize=True).reindex(top_cats, fill_value=0))
        unvoted_counts = (df_cat_top[~df_cat_top["a_recu_vote"]][cat_col]
                          .value_counts(normalize=True).reindex(top_cats, fill_value=0))

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(top_cats))
        width = 0.35
        ax.bar(x - width / 2, voted_counts.values, width, label="Votees", alpha=0.8)
        ax.bar(x + width / 2, unvoted_counts.values, width, label="Non votees", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(top_cats, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Proportion")
        ax.set_title("Exercice 1.2 - Distribution des categories: votees vs non votees (top 15)")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, output_dir / "exercice_1_2_categories.png")

        # Test chi2 sur TOUTES les categories
        contingency = pd.crosstab(df_cat[cat_col], df_cat["a_recu_vote"])
        if contingency.shape[0] >= 2 and contingency.shape[1] == 2:
            chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)
            n_obs = contingency.sum().sum()
            k = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_obs * k)) if k > 0 else 0
            print(f"  categories (toutes, n={contingency.shape[0]}): chi2={chi2:.2f}, p={chi2_p:.2e}, V={cramers_v:.4f}")
            stats_rows.append({
                "Variable": "categories",
                "Moyenne_votees": "-", "Moyenne_non_votees": "-",
                "Mediane_votees": "-", "Mediane_non_votees": "-",
                "Ecart_type_votees": "-", "Ecart_type_non_votees": "-",
                "Mann_Whitney_U": "-", "MW_p_value": "-",
                "KS_stat": f"chi2={chi2:.2f}",
                "KS_p_value": f"{chi2_p:.2e}",
                "Rank_biserial_r": f"V={cramers_v:.4f}",
            })

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(output_dir / "biais_selection_stats.csv", index=False)
    return df_stats


# -----------------------------------------------------------------------
# Exercice 1.3 - alpha de Krippendorff
# -----------------------------------------------------------------------

def _run_krippendorff(df_filtered: pd.DataFrame, visitor_col: str,
                      labels: list[str], threshold_used: int) -> list[dict]:
    """Calcule le alpha de Krippendorff a partir d'un df multi-annote deja filtre."""
    results = []
    for label in labels:
        df_dedup = df_filtered.drop_duplicates(subset=["item_id", visitor_col], keep="first")
        pivot = df_dedup.pivot_table(
            index=visitor_col, columns="item_id",
            values=label, aggfunc="first"
        )
        reliability_matrix = pivot.values.astype(float)

        try:
            alpha = krippendorff.alpha(
                reliability_data=reliability_matrix,
                level_of_measurement="nominal"
            )
        except Exception:
            alpha = float("nan")

        n_annotators = pivot.shape[0]
        n_items = pivot.shape[1]
        completeness = 1.0 - np.isnan(reliability_matrix).mean()

        results.append({
            "Label": label,
            "Alpha_Krippendorff": round(alpha, 4) if not np.isnan(alpha) else float("nan"),
            "N_annotateurs": n_annotators,
            "N_items": n_items,
            "Seuil_min_annotateurs": threshold_used,
            "Completude_matrice": round(completeness, 4),
        })
        print(f"  {label}: alpha={alpha:.4f}, annotateurs={n_annotators}, items={n_items}")
    return results


def exercice_1_3(df_reactions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Calcule le alpha de Krippendorff pour creative, useful et incorrect.

    Deux approches:
    A) Stricte  — (refers_to_model, response_content, len>50): 14 items avec >=3 visiteurs
    B) Proxy    — (refers_to_model, question_content):         176 items avec >=3 visiteurs

    La matrice de fiabilite est construite par pivot annotateur x item.
    """
    if krippendorff is None:
        print("Package krippendorff non disponible.")
        return pd.DataFrame()

    visitor_col = VISITOR_COL
    if visitor_col not in df_reactions.columns:
        candidates = [c for c in df_reactions.columns
                      if "visitor" in c.lower() or "user" in c.lower()]
        if candidates:
            visitor_col = candidates[0]
        else:
            return pd.DataFrame()

    labels = [c for c in ["creative", "useful", "incorrect"] if c in df_reactions.columns]
    if not labels:
        return pd.DataFrame()

    all_results = []
    approach_dfs = {}

    # --- Approche A: stricte (response_content) ---
    if "response_content" in df_reactions.columns:
        group_a = ["refers_to_model", "response_content"]
        df_a, n_a3 = _build_multi_item_df(
            df_reactions, group_a, visitor_col, labels,
            min_visitors=3, response_len_filter=MIN_RESPONSE_LEN
        )
        if n_a3 < 5:
            # Fallback a 2 visiteurs
            df_a, n_a3 = _build_multi_item_df(
                df_reactions, group_a, visitor_col, labels,
                min_visitors=2, response_len_filter=MIN_RESPONSE_LEN
            )
            thresh_a = 2
        else:
            thresh_a = 3
        print(f"[Stricte] Items (model, response_content, len>{MIN_RESPONSE_LEN}) avec >={thresh_a} visiteurs: {n_a3:,}")
        if len(df_a) > 0:
            res_a = _run_krippendorff(df_a, visitor_col, labels, thresh_a)
            for r in res_a:
                r["Approche"] = "stricte_response_content"
            all_results.extend(res_a)
            approach_dfs["Stricte\n(response_content)"] = res_a

    # --- Approche B: proxy (question_content) ---
    if "question_content" in df_reactions.columns:
        group_b = ["refers_to_model", "question_content"]
        df_b3, n_b3 = _build_multi_item_df(
            df_reactions, group_b, visitor_col, labels, min_visitors=3
        )
        df_b2, n_b2 = _build_multi_item_df(
            df_reactions, group_b, visitor_col, labels, min_visitors=2
        )
        if n_b3 >= 20:
            df_b_use, thresh_b, n_b_use = df_b3, 3, n_b3
        else:
            df_b_use, thresh_b, n_b_use = df_b2, 2, n_b2
        print(f"[Proxy]   Items (model, question_content) avec >={thresh_b} visiteurs: {n_b_use:,}")
        if len(df_b_use) > 0:
            res_b = _run_krippendorff(df_b_use, visitor_col, labels, thresh_b)
            for r in res_b:
                r["Approche"] = "proxy_question_content"
            all_results.extend(res_b)
            approach_dfs["Proxy\n(question_content)"] = res_b

    if not all_results:
        return pd.DataFrame()

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_dir / "krippendorff_alpha.csv", index=False)

    # --- Graphique comparatif ---
    if len(approach_dfs) >= 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        approach_names = list(approach_dfs.keys())
        x = np.arange(len(labels))
        width = 0.35 if len(approach_names) == 2 else 0.6
        colors = ["#4878d0", "#ee854a"]

        for k, (approach_name, res_list) in enumerate(approach_dfs.items()):
            alpha_vals = []
            for label in labels:
                match = [r for r in res_list if r["Label"] == label]
                val = match[0]["Alpha_Krippendorff"] if match else float("nan")
                alpha_vals.append(float(val) if not (isinstance(val, float) and np.isnan(val)) else 0)

            offset = (k - (len(approach_names) - 1) / 2) * width
            bars = ax.bar(x + offset, alpha_vals, width, label=approach_name,
                          color=colors[k % len(colors)], alpha=0.85)
            for bar, val in zip(bars, alpha_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("alpha de Krippendorff")
        ax.set_title("Exercice 1.3 - alpha de Krippendorff\ncomparaison approches")
        ax.set_ylim(-0.1, 1.0)
        ax.axhline(y=0.67, color="orange", linestyle="--", alpha=0.5, label="Seuil tentative (0.67)", linewidth=0.9)
        ax.axhline(y=0.80, color="green", linestyle="--", alpha=0.5, label="Seuil acceptable (0.80)", linewidth=0.9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, output_dir / "exercice_1_3_krippendorff.png")

    return df_results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Question 1 - Fiabilite des jugements humains")
    parser.add_argument("--output-dir", type=str, default="outputs/question1")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--reactions-path", type=str, default=HF_REACTIONS)
    parser.add_argument("--votes-path", type=str, default=HF_VOTES)
    parser.add_argument("--conversations-path", type=str, default=HF_CONVERSATIONS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df_reactions = load_parquet_hf(args.reactions_path, args.sample_size)
    df_reactions = ensure_bool(df_reactions, REACTION_BOOL_COLS)
    df_votes = load_parquet_hf(args.votes_path, args.sample_size)
    df_conversations = load_parquet_hf(args.conversations_path, args.sample_size)

    print("\n--- Exercice 1.1 ---")
    exercice_1_1(df_reactions, output_dir)

    print("\n--- Exercice 1.2 ---")
    exercice_1_2(df_conversations, df_votes, output_dir)

    print("\n--- Exercice 1.3 ---")
    exercice_1_3(df_reactions, output_dir)

    print(f"\nDone. Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
