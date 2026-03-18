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
    
    print(f"Minimum comparisons needed to distinguish rank 3 from rank 5 (80% power, alpha=0.05): {int(np.ceil(N))}")

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
        np.fill_diagonal(denom, 1.0) # avoid division issues on diagonal
        
        p_win = pi_col / denom
        p_tie = num_t / denom
        
        ll_win = w_mat * np.log(p_win + 1e-10)
        ll_tie = t_mat * np.log(p_tie + 1e-10)
        
        nll = - (np.sum(ll_win) + 0.5 * np.sum(ll_tie))
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
    df_base_f, valid_models_b = justify_and_filter_N(df_base, min_comparisons=100)
    W_base = build_wins_matrix(df_base_f, valid_models_b)
    
    test_stochastic_transitivity(W_base, top_k=20)
    
    beta_base = fit_bt_mm(W_base)
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
