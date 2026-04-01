import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import requests
import io
import os
import time

def download_parquet(url, cache_path, retries=3, chunk_size=1024*1024):
    """Download a parquet file with chunked streaming and retry logic."""
    if os.path.exists(cache_path):
        print(f"[cache] Loading {cache_path}")
        return pd.read_parquet(cache_path)
    for attempt in range(1, retries + 1):
        try:
            print(f"[download] Attempt {attempt}/{retries}: {url}")
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            buf = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    buf.write(chunk)
            buf.seek(0)
            df = pd.read_parquet(buf)
            df.to_parquet(cache_path)
            print(f"[download] Saved to {cache_path}")
            return df
        except Exception as e:
            print(f"[download] Error on attempt {attempt}: {e}")
            if attempt < retries:
                time.sleep(3)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")

def exercise_3_1():
    print("=== EXERCICE 3.1: Biais de longueur ===\n")
    # Charger le dataset de votes
    df_votes = download_parquet(
        "https://object.data.gouv.fr/ministere-culture/COMPARIA/votes.parquet",
        "cache_votes.parquet"
    )
    
    # Calculer la différence de longueur : on utilise len() sur conversation_a si les tokens manquent
    if 'total_conv_a_output_tokens' in df_votes.columns:
        len_a = df_votes['total_conv_a_output_tokens']
        len_b = df_votes['total_conv_b_output_tokens']
    else:
        len_a = df_votes['conversation_a'].astype(str).str.len()
        len_b = df_votes['conversation_b'].astype(str).str.len()
        
    df_votes['length_diff'] = len_a - len_b
    
    # Filtrer les égalités, on ne garde que les victoires claires de A ou B si possible
    # Le prompt demande la corrélation entre length_diff et (chosen_model_name == model_a_name)
    df_votes['vote_A'] = (df_votes['chosen_model_name'] == df_votes['model_a_name']).astype(int)
    
    # 1. Corrélation
    valid = df_votes[['length_diff', 'vote_A']].dropna()
    corr, pval = stats.spearmanr(valid['length_diff'], valid['vote_A'])
    print(f"[{valid.shape[0]} matchs analysés]")
    print(f"Corrélation de Spearman (length_diff, vote_A): {corr:.4f} (p-value: {pval:.2e})\n")
    
    # 2. Modèle Bradley-Terry par Régression Logistique
    print("Préparation du modèle Bradley-Terry (Régression logistique)...")
    
    # Récupérer tous les modèles uniques
    models = pd.concat([df_votes['model_a_name'], df_votes['model_b_name']]).dropna().unique()
    
    # Construire la matrice X pour Bradley-Terry : 1 si A, -1 si B, 0 sinon
    X_dict = {m: np.zeros(len(df_votes)) for m in models}
    
    for m in models:
        # Vectorized assignment
        mask_A = (df_votes['model_a_name'] == m)
        mask_B = (df_votes['model_b_name'] == m)
        X_dict[m][mask_A] = 1
        X_dict[m][mask_B] = -1
        
    X = pd.DataFrame(X_dict)
    
    # Enlever les ties (égalité "tie", "tie (bothbad)") pour la régression binaire
    strict_mask = df_votes['chosen_model_name'].isin(models)
    X_strict = X[strict_mask].copy()
    y_strict = df_votes.loc[strict_mask, 'vote_A']
    len_strict = df_votes.loc[strict_mask, 'length_diff']
    
    # Normaliser length_diff pour avoir des coefficients comparables
    # Nous utilisons StandardScaler sans scikit pour aller vite
    mean_len = len_strict.mean()
    std_len = len_strict.std()
    X_strict['length_diff_norm'] = (len_strict - mean_len) / std_len
    
    # Pour éviter la multicolinéarité parfaite, on drop un modèle de référence (le premier)
    ref_model = models[0]
    X_base = X_strict.drop(columns=[ref_model, 'length_diff_norm'])
    X_len = X_strict.drop(columns=[ref_model])
    
    # Interpolation na des features si besoin
    valid_idx = X_base.dropna().index.intersection(y_strict.dropna().index)
    X_base = X_base.loc[valid_idx]
    y_final = y_strict.loc[valid_idx]
    X_len = X_len.loc[valid_idx].dropna() # s'assurer que length_diff n'est pas NaN
    y_final_len = y_strict.loc[X_len.index]
    
    # Entraîner BT de base
    clf_base = LogisticRegression(fit_intercept=False, penalty=None, solver='lbfgs')
    clf_base.fit(X_base, y_final)
    
    # Entraîner BT avec covariance (longueur)
    clf_len = LogisticRegression(fit_intercept=False, penalty=None, solver='lbfgs')
    clf_len.fit(X_len, y_final_len)
    
    # Affichage des classements
    scores_base = {ref_model: 0.0}
    scores_base.update(dict(zip(X_base.columns, clf_base.coef_[0])))
    
    scores_len = {ref_model: 0.0}
    scores_len.update(dict(zip(X_len.columns[:-1], clf_len.coef_[0][:-1])))
    coef_length = clf_len.coef_[0][-1]
    
    print("\nClassement BT (Top 5) AVANT correction longueur:")
    df_base = pd.Series(scores_base).sort_values(ascending=False)
    print(df_base.head(5))
    
    print("\nClassement BT (Top 5) APRES correction longueur:")
    df_len = pd.Series(scores_len).sort_values(ascending=False)
    print(df_len.head(5))
    
    print(f"\nCoefficient associé à length_diff (normalisé): {coef_length:.4f}")
    
    print("\nChangements de rang majeurs:")
    rank_base = df_base.rank(ascending=False)
    rank_len = df_len.rank(ascending=False)
    diff_rank = (rank_base - rank_len).abs().sort_values(ascending=False)
    top_diff = diff_rank[diff_rank > 0]
    if len(top_diff) > 0:
        for model in top_diff.head(3).index:
            print(f"- {model}: Rang base = {rank_base[model]:.0f} -> Rang corrigé = {rank_len[model]:.0f}")
    else:
        print("Aucun changement de rang.")


def exercise_3_2():
    print("\n\n=== EXERCICE 3.2: Biais de position A/B ===\n")
    df_react = download_parquet(
        "https://object.data.gouv.fr/ministere-culture/COMPARIA/reactions.parquet",
        "cache_reactions.parquet"
    )
    
    # Qualificatifs cibles
    cols = ['creative', 'useful', 'incorrect']
    if 'liked' in df_react.columns:
        cols.insert(0, 'liked')
    else:
        # Dans comparia-reactions, les votes positifs/négatifs sont des réactions (upvotes)
        if 'upvote' in df_react.columns:
            cols.insert(0, 'upvote')
            
    # Vérifier les colonnes
    cols = [c for c in cols if c in df_react.columns]
    
    for c in cols:
        print(f"--- Biais de position pour '{c}' ---")
        tab = pd.crosstab(df_react['model_pos'], df_react[c])
        print(tab)
        
        # Test du Chi-deux
        # Uniquement si la matrice a bien 2 lignes (A et B) et qu'on compare un booléen
        if tab.shape[0] >= 2 and tab.shape[1] >= 2:
            chi2, pval, dof, ex = stats.chi2_contingency(tab)
            print(f"Chi-deux = {chi2:.4f}, p-value = {pval:.2e}")
            
            # Taux de positivité (True / True+False)
            if True in tab.columns:
                rate_a = tab.loc['a', True] / tab.loc['a'].sum() if 'a' in tab.index else 0
                rate_b = tab.loc['b', True] / tab.loc['b'].sum() if 'b' in tab.index else 0
                print(f"Taux pour A : {rate_a:.2%} | Taux pour B : {rate_b:.2%}")
                diff = rate_a - rate_b
                print(f"Différence absolue : {diff:.2%} (avantage pour {'A' if diff > 0 else 'B'})")
        print("\n")


if __name__ == "__main__":
    exercise_3_1()
    exercise_3_2()
