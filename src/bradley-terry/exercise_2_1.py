import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

from preprocessing import load_data, clean_data, get_base_data, get_creative_data, justify_and_filter_N, build_wins_matrix

def fit_bt_mm(W: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-6) -> pd.Series:
    """
    Fits the Bradley-Terry model using the Minorize-Maximization (MM) algorithm.
    Returns the log-parameters (beta) centered around 0.
    """
    models = W.index
    n = len(models)
    W_mat = W.values
    
    wins = W_mat.sum(axis=1)
    N_mat = W_mat + W_mat.T
    
    pi = np.ones(n) / n
    
    for iteration in range(max_iter):
        pi_prev = pi.copy()
        
        for i in range(n):
            if wins[i] == 0:
                pi[i] = 1e-10
                continue
            
            denom = 0.0
            for j in range(n):
                if i != j and N_mat[i, j] > 0:
                    denom += N_mat[i, j] / (pi_prev[i] + pi_prev[j])
            
            if denom > 0:
                pi[i] = wins[i] / denom
                
        pi = pi / pi.sum()
        if np.max(np.abs(pi - pi_prev)) < tol:
            break
            
    # Convert to log scale (beta) and center
    beta = np.log(np.maximum(pi, 1e-10))
    beta = beta - beta.mean()
    
    return pd.Series(beta, index=models, name="beta").sort_values(ascending=False)

def plot_rank_comparison(beta_global: pd.Series, beta_creative: pd.Series, output_path: str = "results/rank_comparison.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    common_models = beta_global.index.intersection(beta_creative.index)
    
    ranks_g = beta_global.loc[common_models].rank(ascending=False)
    ranks_c = beta_creative.loc[common_models].rank(ascending=False)
    
    rho, pval = spearmanr(ranks_g, ranks_c)
    print(f"\nSpearman correlation between global and creative rankings: {rho:.3f} (p={pval:.3e})")
    
    # Positive shift means the model's rank improved (value went down) in the creativity subset
    shift = ranks_g - ranks_c
    
    plt.figure(figsize=(10, 8))
    plt.scatter(ranks_g, ranks_c, alpha=0.6, color='b')
    
    # Highlight top movers
    top_movers_up = shift.nlargest(5).index
    top_movers_down = shift.nsmallest(5).index
    
    for m in list(top_movers_up) + list(top_movers_down):
        plt.annotate(m, (ranks_g[m], ranks_c[m]), fontsize=9, xytext=(5, 5), textcoords='offset points')
        
    plt.plot([0, len(common_models)], [0, len(common_models)], 'r--', alpha=0.5, label='No shift')
    plt.xlabel('Global Rank (1 is best)')
    plt.ylabel('Creativity Rank (1 is best)')
    plt.title(f'Rank Shifts: Global vs Creativity-Filtered (Spearman rho: {rho:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    print("\n--- Top Models gaining rank in Creativity subset (relative to global) ---")
    print(shift.nlargest(5))
    
    print("\n--- Top Models losing rank in Creativity subset (relative to global) ---")
    print(shift.nsmallest(5))
    
    return rho, shift

if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    
    print("\n[FITTING GLOBAL MODEL]")
    df_base = get_base_data(df_clean)
    df_base_f, valid_models_b = justify_and_filter_N(df_base, min_comparisons=100)
    W_base = build_wins_matrix(df_base_f, valid_models_b)
    
    beta_global = fit_bt_mm(W_base)
    print("\nTop 10 Global Models (beta params):")
    print(beta_global.head(10))
    
    print("\n[FITTING CREATIVITY MODEL]")
    df_creative = get_creative_data(df_clean)
    df_creative_f, valid_models_c = justify_and_filter_N(df_creative, min_comparisons=20)
    W_creative = build_wins_matrix(df_creative_f, valid_models_c)
    
    beta_creative = fit_bt_mm(W_creative)
    print("\nTop 10 Creativity Models (beta params):")
    print(beta_creative.head(10))
    
    plot_rank_comparison(beta_global, beta_creative)
