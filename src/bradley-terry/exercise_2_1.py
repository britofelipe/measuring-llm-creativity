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

def get_bt_standard_errors(W: pd.DataFrame, beta: pd.Series) -> pd.Series:
    """Calculates asymptotic standard errors using the Moore-Penrose pseudo-inverse of the Fisher Information Matrix."""
    models = W.index
    W_mat = W.values
    N_mat = W_mat + W_mat.T
    
    pi = np.exp(beta.values)
    pi_col = pi[:, np.newaxis]
    pi_row = pi[np.newaxis, :]
    
    denom = (pi_col + pi_row)**2
    np.fill_diagonal(denom, 1.0)
    
    H = - N_mat * (pi_col * pi_row) / denom
    np.fill_diagonal(H, 0)
    H_diag = -np.sum(H, axis=1)
    np.fill_diagonal(H, H_diag)
    
    cov = np.linalg.pinv(-H)
    se = np.sqrt(np.abs(np.diag(cov)))
    
    return pd.Series(se, index=models)

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

def plot_top_models_bar(beta_global: pd.Series, se_global: pd.Series, beta_creative: pd.Series, se_creative: pd.Series, output_path: str = "results/top_models_bar.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top15 = beta_global.head(15).index[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(top15))
    width = 0.35
    
    vals_g = beta_global.loc[top15].values
    err_g = se_global.loc[top15].values * 1.96  # 95% Confidence Interval
    
    vals_c = []
    err_c = []
    for m in top15:
        if m in beta_creative:
            vals_c.append(beta_creative[m])
            err_c.append(se_creative[m] * 1.96)
        else:
            vals_c.append(0)
            err_c.append(0)
    
    ax.barh(y_pos - width/2, vals_g, width, xerr=err_g, label='Global Preference (95% CI)', color='#1f77b4', capsize=3)
    ax.barh(y_pos + width/2, vals_c, width, xerr=err_c, label='Creativity Preference (95% CI)', color='#ff7f0e', capsize=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15)
    ax.set_xlabel('Estimated Preference Strength (Beta)')
    ax.set_title('Top 15 Models: Global vs Creativity-Filtered Preference')
    ax.legend()
    plt.grid(axis='x', alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar plot saved to {output_path}")

def plot_beta_shift_movers(beta_global: pd.Series, se_global: pd.Series, beta_creative: pd.Series, se_creative: pd.Series, output_path: str = "results/beta_shift_movers.png"):
    from scipy.stats import zscore
    common = beta_global.index.intersection(beta_creative.index)
    
    # Standardize beta scores to compare across datasets
    z_g = pd.Series(zscore(beta_global[common]), index=common)
    z_c = pd.Series(zscore(beta_creative[common]), index=common)
    
    shift = z_c - z_g
    
    # Get top 8 gainers and top 8 losers
    top_gainers = shift.nlargest(8)
    top_losers = shift.nsmallest(8)
    movers = pd.concat([top_gainers, top_losers]).sort_values()
    
    plt.figure(figsize=(10, 8))
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in movers.values]
    
    # Horizontal bar
    y_pos = np.arange(len(movers))
    plt.barh(y_pos, movers.values, color=colors, alpha=0.8, edgecolor='black')
    plt.yticks(y_pos, movers.index)
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel('Standardized Preference Shift (Z_creative - Z_global)')
    plt.title('Largest Movers in Creativity-Filtered Regime (Beta Shift)')
    plt.grid(axis='x', alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Shift bar plot saved to {output_path}")

if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    
    print("\n[FITTING GLOBAL MODEL]")
    df_base = get_base_data(df_clean)
    df_base_f, valid_models_b = justify_and_filter_N(df_base, min_comparisons=100, regime_name="Global")
    W_base = build_wins_matrix(df_base_f, valid_models_b)
    
    beta_global = fit_bt_mm(W_base)
    se_global = get_bt_standard_errors(W_base, beta_global)
    print("\nTop 10 Global Models (beta params +/- 1.96*SE):")
    for m in beta_global.head(10).index:
        print(f"{m:30} {beta_global[m]:8.4f}  (+/- {se_global[m]*1.96:.4f})")
    
    print("\n[FITTING CREATIVITY MODEL]")
    df_creative = get_creative_data(df_clean)
    df_creative_f, valid_models_c = justify_and_filter_N(df_creative, min_comparisons=20, regime_name="Creative")
    W_creative = build_wins_matrix(df_creative_f, valid_models_c)
    
    beta_creative = fit_bt_mm(W_creative)
    se_creative = get_bt_standard_errors(W_creative, beta_creative)
    print("\nTop 10 Creativity Models (beta params +/- 1.96*SE):")
    for m in beta_creative.head(10).index:
        print(f"{m:30} {beta_creative[m]:8.4f}  (+/- {se_creative[m]*1.96:.4f})")
    
    plot_rank_comparison(beta_global, beta_creative)
    plot_top_models_bar(beta_global, se_global, beta_creative, se_creative)
    plot_beta_shift_movers(beta_global, se_global, beta_creative, se_creative)
