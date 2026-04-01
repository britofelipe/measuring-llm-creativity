import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

from preprocessing import load_data, clean_data, get_base_data, justify_and_filter_N, build_wins_matrix, build_ties_matrix
from exercise_2_1 import fit_bt_mm

def test_stochastic_transitivity(W: pd.DataFrame, top_k: int = 20):
    N_mat = W + W.T
    comps = N_mat.sum(axis=1)
    top_models = comps.nlargest(top_k).index
    
    W_sub = W.loc[top_models, top_models].values
    N_sub = N_mat.loc[top_models, top_models].values
    
    P = np.zeros((top_k, top_k))
    np.divide(W_sub, N_sub, out=P, where=N_sub>0)
    
    wst_violations = 0
    sst_violations = 0
    triplets = 0
    
    for i in range(top_k):
        for j in range(top_k):
            for k in range(top_k):
                if i != j and j != k and i != k:
                    if P[i,j] >= 0.5 and P[j,k] >= 0.5:
                        triplets += 1
                        if P[i,k] < 0.5:
                            wst_violations += 1
                        if P[i,k] < max(P[i,j], P[j,k]):
                            sst_violations += 1
                            
    print(f"\n--- Transitivity Analysis on Top {top_k} Models ---")
    print(f"Total informative triplets tested: {triplets}")
    print(f"Weak Stochastic Transitivity Violations: {wst_violations} ({(wst_violations/triplets)*100:.1f}%)")
    print(f"Strong Stochastic Transitivity Violations: {sst_violations} ({(sst_violations/triplets)*100:.1f}%)")

import matplotlib.pyplot as plt
import os

def power_analysis(beta_3: float, beta_5: float, alpha=0.05, power=0.80):
    pi_3 = np.exp(beta_3)
    pi_5 = np.exp(beta_5)
    p = pi_3 / (pi_3 + pi_5)
    
    print("\n--- Power Analysis ---")
    print(f"P(Rank 3 > Rank 5) = {p:.4f}")
    
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    p0 = 0.5
    N = ((z_alpha * np.sqrt(p0 * (1 - p0)) + z_beta * np.sqrt(p * (1 - p))) / (p - p0))**2
    n_80 = int(np.ceil(N))
    print(f"Minimum comparisons needed to distinguish rank 3 from rank 5 (80% power, alpha=0.05): {n_80}")
    
    from statsmodels.stats.proportion import proportion_effectsize
    import statsmodels.stats.power as smp
    
    n_obs = np.linspace(100, max(10000, n_80 * 1.5), 100)
    effect_size = proportion_effectsize(p, 0.5)
    powers = smp.NormalIndPower().solve_power(effect_size=effect_size, nobs1=n_obs, alpha=0.05, ratio=1.0)
    
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(n_obs, powers, lw=2, color='darkgreen')
    plt.axhline(0.80, color='red', linestyle='--', alpha=0.7, label='80% Power')
    plt.axvline(n_80, color='red', linestyle='--', alpha=0.7)
    plt.annotate(f'N ≈ {n_80}', xy=(n_80, 0.80), xytext=(n_80*1.1, 0.6),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    plt.title(f'Power Curve to distinguish Rank 3 and 5 (p={p:.3f})')
    plt.xlabel('Number of Comparisons (N)')
    plt.ylabel('Statistical Power')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("results/power_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Power curve plot saved to results/power_curve.png")

def plot_confrontation_heatmap(W: pd.DataFrame, output_path: str = "results/confrontation_heatmap.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    W_mat = W.values
    N_mat = W_mat + W_mat.T
    
    with np.errstate(divide='ignore', invalid='ignore'):
        win_rate = np.divide(W_mat, N_mat)
        win_rate[N_mat == 0] = np.nan
        
    sns.heatmap(win_rate, xticklabels=W.columns, yticklabels=W.index, 
                cmap="RdYlGn", center=0.5, cbar_kws={'label': 'Win Rate (Row vs Col)'})
    plt.title("Top-20 Pairwise Confrontation Heatmap")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")

def plot_calibration(W: pd.DataFrame, beta: pd.Series, output_path: str = "results/calibration_plot.png"):
    n = len(W)
    probs_pred = []
    probs_obs = []
    weights = []
    
    W_mat = W.values
    N_mat = W_mat + W_mat.T
    pi = np.exp(beta.values)
    
    for i in range(n):
        for j in range(i+1, n):
            if N_mat[i, j] >= 5: 
                p_hat = pi[i] / (pi[i] + pi[j])
                p_obs = W_mat[i, j] / N_mat[i, j]
                probs_pred.append(p_hat)
                probs_obs.append(p_obs)
                weights.append(N_mat[i, j])
                
                probs_pred.append(1 - p_hat)
                probs_obs.append(1 - p_obs)
                weights.append(N_mat[i, j])
                
    if not probs_pred:
        return
        
    plt.figure(figsize=(8, 8))
    plt.scatter(probs_pred, probs_obs, s=np.array(weights)*0.5, alpha=0.5, color='indigo')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    from scipy.stats import binned_statistic
    try:
        bin_means, bin_edges, _ = binned_statistic(probs_pred, probs_obs, statistic='mean', bins=8, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, bin_means, 'rx-', lw=2, markersize=8, label='Empirical Calibration Curve')
    except:
        pass
        
    plt.title('Bradley-Terry Calibration Plot (Pairs with >= 5 matches)')
    plt.xlabel('Predicted Win Probability')
    plt.ylabel('Observed Win Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration plot saved to {output_path}")

def fit_davidson(W: pd.DataFrame, T: pd.DataFrame):
    models = W.index
    n = len(models)
    w_mat = W.values
    t_mat = T.values
    
    def neg_log_likelihood(params):
        beta = np.zeros(n)
        beta[:-1] = params[:-1]
        nu = np.exp(params[-1])
        pi = np.exp(beta)
        
        pi_col = pi[:, np.newaxis]
        pi_row = pi[np.newaxis, :]
        
        num_t = nu * np.sqrt(pi_col * pi_row)
        denom = pi_col + pi_row + num_t
        np.fill_diagonal(denom, 1.0)
        
        p_win = pi_col / denom
        p_tie = num_t / denom
        
        ll_win = w_mat * np.log(p_win + 1e-10)
        ll_tie = t_mat * np.log(p_tie + 1e-10)
        
        # Add L2 Regularization to prevent perfect separation and bounded scaling
        l2_penalty = 0.05 * np.sum(beta**2)
        
        nll = - (np.sum(ll_win) + 0.5 * np.sum(ll_tie)) + l2_penalty
        return nll

    x0 = np.zeros(n)
    x0[-1] = np.log(0.5)
    
    res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', options={'disp': False})
    
    beta_res = np.zeros(n)
    beta_res[:-1] = res.x[:-1]
    beta_res = beta_res - np.mean(beta_res)
    nu_res = np.exp(res.x[-1])
    
    series_beta = pd.Series(beta_res, index=models).sort_values(ascending=False)
    return series_beta, nu_res

if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    
    df_base = get_base_data(df_clean)
    df_base_f, valid_models_b = justify_and_filter_N(df_base, min_comparisons=100, regime_name="Global")
    W_base = build_wins_matrix(df_base_f, valid_models_b)
    
    test_stochastic_transitivity(W_base, top_k=20)
    
    beta_base = fit_bt_mm(W_base)
    
    # Top 20 for specific diagnostics
    comps = (W_base + W_base.T).sum(axis=1)
    top_20_models = comps.nlargest(20).index
    W_top20 = W_base.loc[top_20_models, top_20_models]
    
    plot_confrontation_heatmap(W_top20)
    plot_calibration(W_top20, beta_base.loc[top_20_models])
    
    rank_3_val = beta_base.iloc[2]
    rank_5_val = beta_base.iloc[4]
    
    print(f"\nGlobal Rank 3: {beta_base.index[2]} (beta={rank_3_val:.3f})")
    print(f"Global Rank 5: {beta_base.index[4]} (beta={rank_5_val:.3f})")
    power_analysis(rank_3_val, rank_5_val)
    
    print("\n[DAVIDSON MODEL WITH TIES]")
    valid_models_all = pd.concat([df_clean['model_a_name'], df_clean['model_b_name']]).value_counts()
    models_100 = valid_models_all[valid_models_all >= 100].index
    
    df_dav = df_clean[df_clean['model_a_name'].isin(models_100) & df_clean['model_b_name'].isin(models_100)]
    W_dav = build_wins_matrix(df_dav, models_100)
    T_dav = build_ties_matrix(df_dav, models_100)
    
    print(f"Davidson Matrix shape: W={W_dav.shape}, T={T_dav.shape}")
    beta_davidson, nu = fit_davidson(W_dav, T_dav)
    
    print(f"\nEstimated Tie Parameter (nu): {nu:.3f}")
    print("\nTop 10 Models (Davidson):")
    print(beta_davidson.head(10))
    
    from scipy.stats import spearmanr
    common = beta_base.index.intersection(beta_davidson.index)
    rho_d, _ = spearmanr(beta_base[common], beta_davidson[common])
    print(f"\nSpearman correlation between Base BT and Davidson: {rho_d:.4f}")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(beta_base[common], beta_davidson[common], alpha=0.6, color='purple')
    
    min_val = min(beta_base[common].min(), beta_davidson[common].min())
    max_val = max(beta_base[common].max(), beta_davidson[common].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
    
    for m in beta_base.head(5).index:
        if m in common:
            plt.annotate(m, (beta_base[m], beta_davidson[m]), fontsize=9, xytext=(5, -5), textcoords='offset points')
            
    plt.title(f'Bradley-Terry vs Davidson Parameters (rho={rho_d:.2f})')
    plt.xlabel('Base BT Beta (No ties)')
    plt.ylabel('Davidson Beta (With ties)')
    plt.grid(True, alpha=0.3)
    plt.savefig("results/davidson_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Davidson comparison plot saved to results/davidson_comparison.png")
