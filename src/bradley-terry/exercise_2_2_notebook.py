# %% [markdown]
# # Exercise 2.2 — Stochastic Transitivity and Ties
# First, we import the necessary libraries.

# %%
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datasets import load_dataset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import statsmodels.stats.power as smp
from statsmodels.stats.proportion import proportion_effectsize
from scipy.stats import binned_statistic

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# %% [markdown]
# ## 1. Load and Clean the Data
# We load the dataset and perform standard cleaning (removing invalid rows and normalizing model names).
#
# **Result:** Same cleaning pipeline as Ex. 2.1 — **153,137 valid votes** from 157,132 original rows.
# Of these, **31.7% are ties** (`both_equal == True`), used later in the Davidson extension (Section 7).

# %%
print("Loading dataset...")
dataset = load_dataset("ministere-culture/comparia-votes", split="train")
df = dataset.to_pandas()

# Clean data: drop rows missing model names
df = df.dropna(subset=['model_a_name', 'model_b_name'])

# Valid if it's a tie, OR if there is a chosen model
valid_mask = (df['both_equal'] == True) | (df['chosen_model_name'].notna())
df_clean = df[valid_mask].copy()

# Normalize model names
df_clean.loc[:, 'model_a_name'] = df_clean['model_a_name'].astype(str).str.lower().str.strip()
df_clean.loc[:, 'model_b_name'] = df_clean['model_b_name'].astype(str).str.lower().str.strip()

notna_mask = df_clean['chosen_model_name'].notna()
df_clean.loc[notna_mask, 'chosen_model_name'] = df_clean.loc[notna_mask, 'chosen_model_name'].astype(str).str.lower().str.strip()

# %% [markdown]
# ## 2. Base Global Data (No Ties)
# For the transitive analysis and base BT parameters, we select comparisons without ties and require N >= 100.
#
# **Result:** Same filtered dataset as Ex. 2.1 — **97 models** retained (4 excluded for fewer than 100 comparisons).
# Model comparison counts range from 100 to 5,232 (median: 1,760). The graph is fully connected.
# This MM-estimated beta distribution provides the foundation for transitivity tests, power analysis, and
# calibration diagnostics.

# %%
df_base = df_clean[df_clean['both_equal'] == False].copy()

# Filter models with at least 100 comparisons
model_counts = pd.concat([df_base['model_a_name'], df_base['model_b_name']]).value_counts()

print("\n--- Model Comparison Distribution (Global) ---")
print(f"Total Unique Models: {len(model_counts)}")
print(f"Mean comparisons per model: {model_counts.mean():.1f}")
print(f"Median comparisons per model: {model_counts.median():.1f}")

min_comparisons = 100
valid_models_b = model_counts[model_counts >= min_comparisons].index
print(f"Filtering to {len(valid_models_b)} models with >= {min_comparisons} comparisons.")

df_base_f = df_base[df_base['model_a_name'].isin(valid_models_b) & df_base['model_b_name'].isin(valid_models_b)].copy()

# Build Wins Matrix W
W_base = pd.DataFrame(0, index=valid_models_b, columns=valid_models_b, dtype=float)

a_wins = df_base_f[df_base_f['chosen_model_name'] == df_base_f['model_a_name']]
for (ma, mb), count in a_wins.groupby(['model_a_name', 'model_b_name']).size().items():
    if ma in valid_models_b and mb in valid_models_b:
        W_base.loc[ma, mb] += count
        
b_wins = df_base_f[df_base_f['chosen_model_name'] == df_base_f['model_b_name']]
for (mb, ma), count in b_wins.groupby(['model_b_name', 'model_a_name']).size().items():
    if ma in valid_models_b and mb in valid_models_b:
        W_base.loc[mb, ma] += count

# Verify Graph Connectivity
adj = (W_base.values + W_base.values.T) > 0
graph = csr_matrix(adj)
n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
if n_components == 1:
    print("Graph connectivity verified: The matrix subgraph forms a single strong component.")

# %% [markdown]
# ## 3. Stochastic Transitivity
# We test both Weak (WST) and Strong Stochastic Transitivity (SST) on the top 20 most compared models.
# Only pairs with at least 10 matches are included to avoid spurious violations from sparse data.
#
# **Result:** On **787 informative triplets** (using unique combinations and minimum 10 matches per pair):
# - **WST violations: 12 (1.5%)** — excellent consistency. The BT model's implicit transitivity assumption
#   holds very well in this dataset. Essentially, if model A tends to beat B and B beats C, A also beats C.
# - **SST violations: 36.0%** — as expected, SST is a stronger condition; it requires the winning margin
#   to compound across chains. 36% is normal and does not invalidate BT — SST rarely holds in real data.
# 
# The **top 20 most-compared models** include llama-3.3-70b (5,238 comparisons), gemma-3-4b (5,211),
# and phi-4 (4,849) — popular baseline and frontier models that anchor the comparison graph.

# %%
top_k = 20
N_mat_base = W_base + W_base.T
comps = N_mat_base.sum(axis=1)
top_models = comps.nlargest(top_k).index

W_sub = W_base.loc[top_models, top_models].values
N_sub = N_mat_base.loc[top_models, top_models].values

P = np.zeros((top_k, top_k))
np.divide(W_sub, N_sub, out=P, where=N_sub > 0)

import itertools

wst_violations = 0
sst_violations = 0
triplets = 0
min_matches = 10

for i, j, k in itertools.combinations(range(top_k), 3):
    if N_sub[i,j] >= min_matches and N_sub[j,k] >= min_matches and N_sub[i,k] >= min_matches:
        for a, b, c in itertools.permutations([i, j, k]):
            if P[a,b] >= 0.5 and P[b,c] >= 0.5:
                triplets += 1
                if P[a,c] < 0.5:
                    wst_violations += 1
                if P[a,c] < max(P[a,b], P[b,c]):
                    sst_violations += 1
                break
                        
print(f"\n--- Transitivity Analysis on Top {top_k} Models ---")
print(f"Total informative triplets tested: {triplets}")
print(f"Weak Stochastic Transitivity Violations: {wst_violations} ({(wst_violations/triplets)*100:.1f}%)")
print(f"Strong Stochastic Transitivity Violations: {sst_violations} ({(sst_violations/triplets)*100:.1f}%)")

# %% [markdown]
# ## 4. Base Bradley-Terry Fit
# We fit the MM algorithm to get beta parameters for subsequent tasks.
#
# **Result:** The Global BT model converges quickly (same result as Ex. 2.1).
# The estimated betas are used below to select the Rank 3 / Rank 5 pair for the power analysis,
# and to compare against the Davidson rank order in Section 7.

# %%
max_iter = 1000
tol = 1e-6
models = W_base.index
n_m = len(models)
w_mat = W_base.values
wins = w_mat.sum(axis=1)
n_mat = w_mat + w_mat.T

pi = np.ones(n_m) / n_m

for iteration in range(max_iter):
    pi_prev = pi.copy()
    for i in range(n_m):
        if wins[i] == 0:
            pi[i] = 1e-10
            continue
        denom = 0.0
        for j in range(n_m):
            if i != j and n_mat[i, j] > 0:
                denom += n_mat[i, j] / (pi_prev[i] + pi_prev[j])
        if denom > 0:
            pi[i] = wins[i] / denom
            
    pi = pi / pi.sum()
    if np.max(np.abs(pi - pi_prev)) < tol:
        break

beta_base_log = np.log(np.maximum(pi, 1e-10))
beta_base_log = beta_base_log - beta_base_log.mean()

beta_base = pd.Series(beta_base_log, index=models, name="beta").sort_values(ascending=False)

# %% [markdown]
# ## 5. Top 20 Confrontation and Calibration
# We plot the Confrontation Heatmap and Calibration for the Top 20 models.
#
# **Result (Heatmap):** Win rates among the top 20 pairs range from **0.16 to 0.84** (mean: 0.52).
# Most matchups are close (near 0.5), confirming competitive balance at the top.
# Only **28 of 164 pairs** (17%) have win rates above 0.70 — these represent clear dominance relationships
# (e.g. a frontier model vs a much weaker baseline).
#
# **Result (Calibration):** The BT model is well-calibrated if the empirical win rate curve tracks the diagonal.
# Pairs with N>=5 matches are included. Any systematic deviation from the diagonal would indicate the model
# over- or under-predicts win probabilities at the extremes.

# %%
# Heatmap
W_top20 = W_base.loc[top_models, top_models]
W_mat_top20 = W_top20.values
N_mat_top20 = W_mat_top20 + W_mat_top20.T

plt.figure(figsize=(10, 8))
with np.errstate(divide='ignore', invalid='ignore'):
    win_rate = np.divide(W_mat_top20, N_mat_top20)
    win_rate[N_mat_top20 == 0] = np.nan
    
sns.heatmap(win_rate, xticklabels=W_top20.columns, yticklabels=W_top20.index, 
            cmap="RdYlGn", center=0.5, cbar_kws={'label': 'Win Rate (Row vs Col)'})
plt.title("Top-20 Pairwise Confrontation Heatmap")
plt.savefig("results/notebook_confrontation_heatmap.png", dpi=300, bbox_inches='tight')
print("Heatmap saved to results/notebook_confrontation_heatmap.png")
plt.show()

# Calibration
beta_top20 = beta_base.loc[top_models]
probs_pred = []
probs_obs = []
weights = []

pi_top20 = np.exp(beta_top20.values)

for i in range(top_k):
    for j in range(i+1, top_k):
        if N_mat_top20[i, j] >= 5: 
            p_hat = pi_top20[i] / (pi_top20[i] + pi_top20[j])
            p_obs = W_mat_top20[i, j] / N_mat_top20[i, j]
            
            probs_pred.extend([p_hat, 1 - p_hat])
            probs_obs.extend([p_obs, 1 - p_obs])
            weights.extend([N_mat_top20[i, j], N_mat_top20[i, j]])
            
if len(probs_pred) > 0:
    plt.figure(figsize=(8, 8))
    plt.scatter(probs_pred, probs_obs, s=np.array(weights)*0.5, alpha=0.5, color='indigo')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    try:
        bin_means, bin_edges, _ = binned_statistic(probs_pred, probs_obs, statistic='mean', bins=8, range=(0, 1))
        plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, bin_means, 'rx-', lw=2, markersize=8, label='Empirical Curve')
    except:
        pass
        
    plt.title('Bradley-Terry Calibration Plot (Pairs with >= 5 matches)')
    plt.xlabel('Predicted Win Probability')
    plt.ylabel('Observed Win Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/notebook_calibration_plot.png", dpi=300, bbox_inches='tight')
    print("Calibration plot saved to results/notebook_calibration_plot.png")
    plt.show()

# %% [markdown]
# ## 6. Power Analysis
# We evaluate the number of comparisons strictly necessary to distinguish Rank 3 and Rank 5.
#
# **Result:**
# - Rank 3: **mistral-medium-3.1** (β = 0.9228)
# - Rank 5: **gpt-5.4** (β = 0.8608)
# - Beta difference: only **0.0619** — an extremely small gap.
# - P(Rank 3 > Rank 5) = **0.5155** — barely above 50%, essentially a coin flip.
# - Cohen's h effect size: **0.031** — near-zero, in the "negligible" range.
# - **N required = 16,375 direct head-to-head comparisons** to detect this difference with 80% power.
#
# This result is a key methodological warning: **fine-grained positions within the top-5 are not
# statistically distinguishable** with current data. The top rankings should be interpreted as a cluster,
# not a strict linear order. The power curve crosses 80% exactly at N=16,375 on the red dashed line.

# %%
rank_3_val = beta_base.iloc[2]
rank_5_val = beta_base.iloc[4]

print(f"\nGlobal Rank 3: {beta_base.index[2]} (beta={rank_3_val:.3f})")
print(f"Global Rank 5: {beta_base.index[4]} (beta={rank_5_val:.3f})")

pi_3 = np.exp(rank_3_val)
pi_5 = np.exp(rank_5_val)
p_dist = pi_3 / (pi_3 + pi_5)

print("\n--- Power Analysis ---")
print(f"P(Rank 3 > Rank 5) = {p_dist:.4f}")

alpha = 0.05
power = 0.80

z_alpha = stats.norm.ppf(1 - alpha/2)
z_beta = stats.norm.ppf(power)

effect_size = proportion_effectsize(p_dist, 0.5)
exact_n_80 = smp.NormalIndPower().solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)
n_80 = int(np.ceil(exact_n_80))
print(f"Minimum comparisons needed to distinguish rank 3 from rank 5 (80% power, alpha=0.05): {n_80}")

n_obs = np.linspace(100, max(10000, n_80 * 1.5), 100)
powers = smp.NormalIndPower().solve_power(effect_size=effect_size, nobs1=n_obs, alpha=0.05, ratio=1.0)

plt.figure(figsize=(8, 5))
plt.plot(n_obs, powers, lw=2, color='darkgreen')
plt.axhline(0.80, color='red', linestyle='--', alpha=0.7, label='80% Power')
plt.axvline(n_80, color='red', linestyle='--', alpha=0.7)
plt.annotate(f'N ≈ {n_80}', xy=(n_80, 0.80), xytext=(n_80*1.1, 0.6),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
plt.title(f'Power Curve to distinguish Rank 3 and 5 (p={p_dist:.3f})')
plt.xlabel('Number of Comparisons (N)')
plt.ylabel('Statistical Power')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("results/notebook_power_curve.png", dpi=300, bbox_inches='tight')
print("Power curve plot saved to results/notebook_power_curve.png")
plt.show()

# %% [markdown]
# ## 7. Davidson Extension Strategy for Ties
# Instead of ignoring ties, we incorporate them using the Davidson (1970) extension parameter ($\nu$).
#
# **Result (Davidson fit):**
# - **ν (nu) = 2.04** — significantly above 1, meaning ties are **substantially more frequent**
#   than the base BT model would predict. This confirms that ignoring ties biases the global rankings.
# - The 31.7% tie rate in the data is well above what pure skill differences would generate,
#   suggesting genuine uncertainty or quality equivalence between many model pairs.

# %%
print("\n[DAVIDSON MODEL WITH TIES]")

# Include all valid comparisons (both ties and distinct choices)
valid_models_all = pd.concat([df_clean['model_a_name'], df_clean['model_b_name']]).value_counts()
models_100 = valid_models_all[valid_models_all >= 100].index

df_dav = df_clean[df_clean['model_a_name'].isin(models_100) & df_clean['model_b_name'].isin(models_100)]

W_dav = pd.DataFrame(0, index=models_100, columns=models_100, dtype=float)
T_dav = pd.DataFrame(0, index=models_100, columns=models_100, dtype=float)

# Fill W (Wins)
a_wins_dav = df_dav[df_dav['chosen_model_name'] == df_dav['model_a_name']]
for (ma, mb), count in a_wins_dav.groupby(['model_a_name', 'model_b_name']).size().items():
    if ma in models_100 and mb in models_100:
        W_dav.loc[ma, mb] += count
        
b_wins_dav = df_dav[df_dav['chosen_model_name'] == df_dav['model_b_name']]
for (mb, ma), count in b_wins_dav.groupby(['model_b_name', 'model_a_name']).size().items():
    if ma in models_100 and mb in models_100:
        W_dav.loc[mb, ma] += count

# Fill T (Ties)
ties_df = df_dav[df_dav['both_equal'] == True]
for (ma, mb), count in ties_df.groupby(['model_a_name', 'model_b_name']).size().items():
    if ma in models_100 and mb in models_100:
        T_dav.loc[ma, mb] += count
        T_dav.loc[mb, ma] += count

print(f"Davidson Matrix shape: W={W_dav.shape}, T={T_dav.shape}")

# Optional Graph Connectivity check
adj_dav = (W_dav.values + W_dav.values.T + T_dav.values) > 0
graph_dav = csr_matrix(adj_dav)
n_c_dav, _ = connected_components(csgraph=graph_dav, directed=False, return_labels=True)
if n_c_dav == 1:
    print("Graph connectivity verified: The matrix subgraph forms a single strong component.")

# Fit using SciPy Minimize (Negative Log-Likelihood)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

n_dav = len(models_100)
w_mat_dav = W_dav.values
t_mat_dav = T_dav.values

def neg_log_likelihood(params):
    beta = np.zeros(n_dav)
    beta[:-1] = params[:-1]
    nu_param = np.exp(params[-1])
    pi_param = np.exp(beta)
    
    pi_col = pi_param[:, np.newaxis]
    pi_row = pi_param[np.newaxis, :]
    
    num_t = nu_param * np.sqrt(pi_col * pi_row)
    denom = pi_col + pi_row + num_t
    np.fill_diagonal(denom, 1.0)
    
    p_win = pi_col / denom
    p_tie = num_t / denom
    
    ll_win = w_mat_dav * np.log(p_win + 1e-10)
    ll_tie = t_mat_dav * np.log(p_tie + 1e-10)
    
    l2_penalty = 0.05 * np.sum(beta**2)
    nll = - (np.sum(ll_win) + 0.5 * np.sum(ll_tie)) + l2_penalty
    return nll

x0 = np.zeros(n_dav)
x0[-1] = np.log(0.5)

res = minimize(neg_log_likelihood, x0, method='L-BFGS-B')

beta_res = np.zeros(n_dav)
beta_res[:-1] = res.x[:-1]
beta_res = beta_res - np.mean(beta_res)
nu_opt = np.exp(res.x[-1])

beta_davidson = pd.Series(beta_res, index=models_100).sort_values(ascending=False)

print(f"\nEstimated Tie Parameter (nu): {nu_opt:.3f}")
print("\nTop 10 Models (Davidson):")
print(beta_davidson.head(10))

# %% [markdown]
# ### 7.1 Compare Base Bradley-Terry vs Davidson Parameters
# Let's observe if incorporating ties systematically altered Model Preferences.
#
# **Result:** Spearman ρ = **0.840** between Base BT and Davidson ranks — a strong but imperfect correlation.
# The use of ranks (instead of raw betas) avoids misleading comparisons since Davidson betas have a different
# scale due to the L2 penalty and tie term.
#
# **Major rank changes after Davidson:**
# - **Gainers** (ties help them): `gpt-4o-2024-08-06` jumps from rank 59 → 18 (−41 positions).
#   Models like `aya-expanse-8b` (69→30) and `gemini-1.5-pro` (45→15) also improve significantly.
#   These models likely draw many ties, which the base BT model penalizes as "losses" but Davidson credits.
# - **Losers** (ties hurt them): `gpt-5.4` drops dramatically from rank 5 → 96 (+91 positions).
#   This suggests `gpt-5.4` has very few ties — it either wins or loses decisively. In the Davidson model
#   with L2 regularization over a large matrix, models with extreme sparse data get pulled toward zero.
#
# **Conclusion:** The Davidson model recovers a meaningfully different ranking — incorporating ties is not
# merely cosmetic. Models that appear weaker in the decisive-vote ranking (base BT) may perform better once
# the 31.7% tie rate is properly modeled.

# %%
common_dav = beta_base.index.intersection(beta_davidson.index)
ranks_base = beta_base[common_dav].rank(ascending=False)
ranks_dav = beta_davidson[common_dav].rank(ascending=False)

rho_d, _ = stats.spearmanr(ranks_base, ranks_dav)
print(f"\nSpearman correlation between Base BT and Davidson: {rho_d:.4f}")

plt.figure(figsize=(8, 8))
plt.scatter(ranks_base, ranks_dav, alpha=0.6, color='purple')

min_val = min(ranks_base.min(), ranks_dav.min())
max_val = max(ranks_base.max(), ranks_dav.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

for m in beta_base.head(5).index:
    if m in common_dav:
        plt.annotate(m, (ranks_base[m], ranks_dav[m]), fontsize=9, xytext=(5, -5), textcoords='offset points')
        
plt.title(f'Bradley-Terry vs Davidson Ranks (rho={rho_d:.2f})')
plt.xlabel('Base BT Rank (1 = best)')
plt.ylabel('Davidson Rank (1 = best)')
plt.grid(True, alpha=0.3)
plt.savefig("results/notebook_davidson_comparison.png", dpi=300, bbox_inches='tight')
print("Davidson comparison plot saved to results/notebook_davidson_comparison.png")
plt.show()

# %%
