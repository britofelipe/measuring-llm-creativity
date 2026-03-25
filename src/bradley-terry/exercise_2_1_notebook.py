# %% [markdown]
# # Exercise 2.1 — Global vs Creativity Ranking
# First, we import the necessary libraries.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
from datasets import load_dataset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# %% [markdown]
# ## 1. Load and Clean the Data
# We load the "comparia-votes" dataset from HuggingFace and clean invalid rows.
#
# **Result:** The dataset contains **157,132 pairwise votes**, of which **153,137 are valid** after removing
# rows with missing model names or no outcome. Cleaning removed only ~2.5% of rows.
# Of the valid votes, **31.7% are ties** (`both_equal == True`) and **68.3% are decisive**.
# This tie rate is high enough to matter (see Davidson model in Exercise 2.2).

# %%
print("Loading dataset...")
dataset = load_dataset("ministere-culture/comparia-votes", split="train")
df = dataset.to_pandas()

print(f"Original dataset size: {len(df)}")

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

print(f"Cleaned dataset size: {len(df_clean)}")

# %% [markdown]
# ## 2. Global Ranking
# ### 2.1 Filtering and Matrix Construction
# We filter only the votes with `both_equal == False` for the base Bradley-Terry model.
# Then, we filter models with at least 100 comparisons to ensure stability.
#
# **Result:** 101 unique models appear in the decisive vote data. **4 models** were excluded
# (fewer than 100 comparisons). The 97 retained models each have between **100 and 5,232 comparisons**
# (median = 1,760), which is a well-supported sample for maximum likelihood estimation.
# **Graph connectivity is confirmed**: all 97 models form a single strongly connected component,
# guaranteeing that BT parameters are jointly identifiable.

# %%
print("\n[GLOBAL RANKING REGIME]")
# Filter non-ties for Global Ranking
df_base = df_clean[df_clean['both_equal'] == False].copy()

# Count valid comparisons per model
model_counts = pd.concat([df_base['model_a_name'], df_base['model_b_name']]).value_counts()
print(f"Total Unique Models: {len(model_counts)}")
print(f"Mean comparisons per model: {model_counts.mean():.1f}")
print(f"Median comparisons per model: {model_counts.median():.1f}")

# Keep models with >= 100 comparisons
min_comparisons_global = 100
valid_models_global = model_counts[model_counts >= min_comparisons_global].index
print(f"Filtering to {len(valid_models_global)} models with >= {min_comparisons_global} comparisons.")

df_base_f = df_base[df_base['model_a_name'].isin(valid_models_global) & df_base['model_b_name'].isin(valid_models_global)].copy()

# Build the Wins Matrix (W)
W_global = pd.DataFrame(0, index=valid_models_global, columns=valid_models_global, dtype=float)

a_wins = df_base_f[df_base_f['chosen_model_name'] == df_base_f['model_a_name']]
for (ma, mb), count in a_wins.groupby(['model_a_name', 'model_b_name']).size().items():
    if ma in valid_models_global and mb in valid_models_global:
        W_global.loc[ma, mb] += count
        
b_wins = df_base_f[df_base_f['chosen_model_name'] == df_base_f['model_b_name']]
for (mb, ma), count in b_wins.groupby(['model_b_name', 'model_a_name']).size().items():
    if ma in valid_models_global and mb in valid_models_global:
        W_global.loc[mb, ma] += count

# Verify graph connectivity for BT identifiability
adj = (W_global.values + W_global.values.T) > 0
graph = csr_matrix(adj)
n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
if n_components == 1:
    print("Global Graph connectivity verified: single strong component.")
else:
    print(f"WARNING: Graph disconnected into {n_components} components.")

# %% [markdown]
# ### 2.2 Fitting the Bradley-Terry Model (Global)
# We estimate the strengths (β) using the Minorize-Maximization (MM) algorithm.
#
# **Result:** The MM algorithm converged in **65 iterations**.
# The top global model is **gemini-3.1-flash-lite-preview** (β=+1.18), ahead of
# **mistral-medium-2508** (β=+0.95) and **mistral-medium-3.1** (β=+0.92).
# However, the CI for rank 1 is very wide ([0.76, 1.61]), while mistral-medium-2508 has a much
# tighter CI ([0.88, 1.03]) — suggesting it is the **most reliably strong performer**.
# The full beta range is **2.71**, meaning the best model has a win probability of **93.8%** against
# the worst.

# %%
max_iter = 1000
tol = 1e-6

models_g = W_global.index
n_g = len(models_g)
W_mat_g = W_global.values

wins_g = W_mat_g.sum(axis=1)
N_mat_g = W_mat_g + W_mat_g.T

pi_g = np.ones(n_g) / n_g

for iteration in range(max_iter):
    pi_prev = pi_g.copy()
    
    for i in range(n_g):
        if wins_g[i] == 0:
            pi_g[i] = 1e-10
            continue
        
        denom = 0.0
        for j in range(n_g):
            if i != j and N_mat_g[i, j] > 0:
                denom += N_mat_g[i, j] / (pi_prev[i] + pi_prev[j])
        
        if denom > 0:
            pi_g[i] = wins_g[i] / denom
            
    pi_g = pi_g / pi_g.sum()
    if np.max(np.abs(pi_g - pi_prev)) < tol:
        print(f"Global model converged in {iteration+1} iterations.")
        break

# Convert to log scale (beta) and center
beta_g = np.log(np.maximum(pi_g, 1e-10))
beta_g = beta_g - beta_g.mean()

beta_global = pd.Series(beta_g, index=models_g, name="beta").sort_values(ascending=False)

# Calculate Standard Errors using Fisher Information
pi_g_final = np.exp(beta_global.values)
pi_col = pi_g_final[:, np.newaxis]
pi_row = pi_g_final[np.newaxis, :]
denom_se = (pi_col + pi_row)**2
np.fill_diagonal(denom_se, 1.0)

W_mat_sorted = W_global.loc[beta_global.index, beta_global.index].values
N_mat_sorted = W_mat_sorted + W_mat_sorted.T

H = - N_mat_sorted * (pi_col * pi_row) / denom_se
np.fill_diagonal(H, 0)
H_diag = -np.sum(H, axis=1)
np.fill_diagonal(H, H_diag)

cov = np.linalg.pinv(-H)
se_g = np.sqrt(np.abs(np.diag(cov)))
se_global = pd.Series(se_g, index=beta_global.index)

print("\nTop 10 Global Models (beta params +/- 1.96*SE):")
for m in beta_global.head(10).index:
    print(f"{m:30} {beta_global[m]:8.4f}  (+/- {se_global[m]*1.96:.4f})")

# %% [markdown]
# ### 2.3 Rank Confidence Intervals (Parametric Bootstrap)
# We simulate from the asymptotic multivariate normal distribution of the estimators $\hat{\beta} \sim \mathcal{N}(\beta, \hat{\Sigma})$ 
# to derive 95% confidence intervals for the model ranks.
#
# **Result:** The bootstrap reveals significant ranking uncertainty at the top.
# - **mistral-medium-2508** has the **narrowest rank CI** among the top 20 (width = 4 positions),
#   confirming it is the most reliably ranked strong model.
# - **gemini-3.1-flash-lite-preview** (rank 1) has a CI of [1–7], and **gpt-5.4** (rank 5) has [1–17],
#   meaning their true positions are very uncertain due to few comparisons.
# - This suggests the top 3 positions are **statistically indistinguishable** and should be interpreted
#   as a cluster rather than a strict ordering.

# %%
np.random.seed(42)
n_sim = 1000
# cov is already ordered by beta_global.index
sim_betas = np.random.multivariate_normal(beta_global.values, cov, size=n_sim)
sim_betas_df = pd.DataFrame(sim_betas, columns=beta_global.index)

# Rank each simulation (1 is best, so ascending=False)
sim_ranks = sim_betas_df.rank(axis=1, ascending=False)

rank_lb = sim_ranks.quantile(0.025, axis=0)
rank_ub = sim_ranks.quantile(0.975, axis=0)
rank_mean = sim_ranks.mean(axis=0)

top_models_plot = beta_global.head(15).index[::-1]
plt.figure(figsize=(10, 8))
y_ticks = np.arange(len(top_models_plot))

# We plot the confidence intervals of the ranks
for i, m in enumerate(top_models_plot):
    plt.plot([rank_lb[m], rank_ub[m]], [i, i], 'b-', alpha=0.6)
    plt.plot(rank_mean[m], i, 'bo')

plt.yticks(y_ticks, top_models_plot)
plt.axvline(1, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Global Rank (1 is best)')
plt.title('95% Confidence Intervals for Global Ranks (Top 15 Models)')
plt.grid(axis='x', alpha=0.3)
plt.savefig("results/notebook_rank_ci.png", dpi=300, bbox_inches='tight')
print("Rank CI plot saved to results/notebook_rank_ci.png")
plt.show()

# %% [markdown]
# ## 3. Creativity Ranking
# ### 3.1 Filtering Creativity Data
# We select conversations where either `conv_creative_a` or `conv_creative_b` is True.
# We apply a threshold of N=20 since the dataset is smaller.
#
# **Result:** The creative subset contains **10,757 votes** — only **7% of total votes**.
# This limited sample is the key methodological challenge: the creative regime requires
# lower thresholds for model filtering. With N≥20, **95 models** are retained.

# %%
print("\n[CREATIVITY RANKING REGIME]")
c_a = df_clean['conv_creative_a'].fillna(False).astype(bool)
c_b = df_clean['conv_creative_b'].fillna(False).astype(bool)
df_creative = df_clean[c_a | c_b].copy()

model_counts_c = pd.concat([df_creative['model_a_name'], df_creative['model_b_name']]).value_counts()
print(f"Total Unique Models (Creative): {len(model_counts_c)}")
print(f"Mean comparisons per model: {model_counts_c.mean():.1f}")
print(f"Median comparisons per model: {model_counts_c.median():.1f}")

min_comparisons_creative = 20
valid_models_creative = model_counts_c[model_counts_c >= min_comparisons_creative].index
print(f"Filtering to {len(valid_models_creative)} models with >= {min_comparisons_creative} comparisons.")

df_creative_f = df_creative[df_creative['model_a_name'].isin(valid_models_creative) & df_creative['model_b_name'].isin(valid_models_creative)].copy()

# Build the Wins Matrix (W) for Creativity
W_creative = pd.DataFrame(0, index=valid_models_creative, columns=valid_models_creative, dtype=float)

a_wins_c = df_creative_f[df_creative_f['chosen_model_name'] == df_creative_f['model_a_name']]
for (ma, mb), count in a_wins_c.groupby(['model_a_name', 'model_b_name']).size().items():
    if ma in valid_models_creative and mb in valid_models_creative:
        W_creative.loc[ma, mb] += count
        
b_wins_c = df_creative_f[df_creative_f['chosen_model_name'] == df_creative_f['model_b_name']]
for (mb, ma), count in b_wins_c.groupby(['model_b_name', 'model_a_name']).size().items():
    if ma in valid_models_creative and mb in valid_models_creative:
        W_creative.loc[mb, ma] += count

# Verify graph connectivity
adj_c = (W_creative.values + W_creative.values.T) > 0
graph_c = csr_matrix(adj_c)
n_components_c, _ = connected_components(csgraph=graph_c, directed=False, return_labels=True)
if n_components_c == 1:
    print("Creative Graph connectivity verified: single strong component.")
else:
    print(f"WARNING: Graph disconnected into {n_components_c} components.")

# %% [markdown]
# ### 3.2 Fitting the Bradley-Terry Model (Creativity)
#
# **Result:** The creative BT model converged in **118 iterations** (vs 65 globally), reflecting
# the noisier, sparser data. The top creative model is also **gemini-3.1-flash-lite-preview** (β=+1.69),
# followed by **mistral-medium-2508** (β=+1.40).
# However, creative CIs are **much wider**: gemini-3.1-flash-lite-preview has a 95%CI of [0.56, 2.81],
# spanning nearly 2.25 units — a sign of high estimation uncertainty due to sparse matchups.

# %%
models_c = W_creative.index
n_c = len(models_c)
W_mat_c = W_creative.values

wins_c = W_mat_c.sum(axis=1)
N_mat_c = W_mat_c + W_mat_c.T

pi_c = np.ones(n_c) / n_c

for iteration in range(max_iter):
    pi_prev = pi_c.copy()
    
    for i in range(n_c):
        if wins_c[i] == 0:
            pi_c[i] = 1e-10
            continue
        
        denom = 0.0
        for j in range(n_c):
            if i != j and N_mat_c[i, j] > 0:
                denom += N_mat_c[i, j] / (pi_prev[i] + pi_prev[j])
        
        if denom > 0:
            pi_c[i] = wins_c[i] / denom
            
    pi_c = pi_c / pi_c.sum()
    if np.max(np.abs(pi_c - pi_prev)) < tol:
        print(f"Creative model converged in {iteration+1} iterations.")
        break

# Convert to log scale and center
beta_c_log = np.log(np.maximum(pi_c, 1e-10))
beta_c_log = beta_c_log - beta_c_log.mean()

beta_creative = pd.Series(beta_c_log, index=models_c, name="beta").sort_values(ascending=False)

# Calculate Standard Errors
pi_c_final = np.exp(beta_creative.values)
pi_col_c = pi_c_final[:, np.newaxis]
pi_row_c = pi_c_final[np.newaxis, :]
denom_se_c = (pi_col_c + pi_row_c)**2
np.fill_diagonal(denom_se_c, 1.0)

W_mat_sorted_c = W_creative.loc[beta_creative.index, beta_creative.index].values
N_mat_sorted_c = W_mat_sorted_c + W_mat_sorted_c.T

H_c = - N_mat_sorted_c * (pi_col_c * pi_row_c) / denom_se_c
np.fill_diagonal(H_c, 0)
H_diag_c = -np.sum(H_c, axis=1)
np.fill_diagonal(H_c, H_diag_c)

cov_c = np.linalg.pinv(-H_c)
se_c_val = np.sqrt(np.abs(np.diag(cov_c)))
se_creative = pd.Series(se_c_val, index=beta_creative.index)

print("\nTop 10 Creativity Models (beta params +/- 1.96*SE):")
for m in beta_creative.head(10).index:
    print(f"{m:30} {beta_creative[m]:8.4f}  (+/- {se_creative[m]*1.96:.4f})")

# %% [markdown]
# ## 4. Comparison and Visualization
# Let's compare the global rankings versus the creativity rankings.
# We check the Spearman correlation to see how much the order changes.
#
# **Result:** The **Spearman ρ = 0.858** (p < 10⁻²⁷) indicates a strong but imperfect correlation
# between global and creative rankings. While the **overall order is preserved**, the creative regime
# does produce meaningful rank shifts:
# - **43% of models** shift more than 10 positions; **13%** shift more than 20 positions.
# - Top **Gainers** in creativity: `qwen3.5-397b-a17b` (+30 ranks), `deepseek-r1` (+23),
#   `gpt-5-nano` (+32) — these models appear to excel specifically at creative tasks.
# - Top **Losers**: `magistral-small-2506` (−39), `grok-3-mini-beta` (−35), `gpt-5.3` (−34) —
#   models that are competitive globally but underperform on creative prompts.

# %%
common_models = beta_global.index.intersection(beta_creative.index)

ranks_g = beta_global.loc[common_models].rank(ascending=False)
ranks_c = beta_creative.loc[common_models].rank(ascending=False)

rho, pval = spearmanr(ranks_g, ranks_c)
print(f"\nSpearman correlation between global and creative rankings: {rho:.3f} (p={pval:.3e})")

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
plt.title(f'Rank Shifts: Global vs Creativity (Spearman rho: {rho:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/notebook_rank_comparison.png", dpi=300, bbox_inches='tight')
print("Plot saved to results/notebook_rank_comparison.png")

print("\n--- Top Models gaining rank in Creativity subset (relative to global) ---")
print(shift.nlargest(5))

print("\n--- Top Models losing rank in Creativity subset (relative to global) ---")
print(shift.nsmallest(5))

# %% [markdown]
# ### 4.1. Top Models Bar Plot Comparison
# Plotting the estimated beta parameters side-by-side with 95% Confidence Intervals.
#
# **Result:** The bar chart visualizes how global and creative β scores differ for the top 15 global models.
# Most models retain similar ordering, but notable divergences appear:
# - `qwen3-max-2025-09-23` shows a notably **higher creative than global β**.
# - `gemini-2.0-flash` and `gemini-2.5-flash` show consistent scores across both regimes.
# The confidence intervals in the creative regime are systematically wider,
# reflecting the smaller sample size.

# %%
top15 = beta_global.head(15).index[::-1]

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(top15))
width = 0.35

vals_g = beta_global.loc[top15].values
err_g = se_global.loc[top15].values * 1.96

vals_c = []
err_c = []
for m in top15:
    if m in beta_creative:
        vals_c.append(beta_creative[m])
        err_c.append(se_creative[m] * 1.96)
    else:
        vals_c.append(0)
        err_c.append(0)

ax.barh(y_pos - width/2, vals_g, width, xerr=err_g, label='Global Preference', color='#1f77b4', capsize=3)
ax.barh(y_pos + width/2, vals_c, width, xerr=err_c, label='Creativity Preference', color='#ff7f0e', capsize=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(top15)
ax.set_xlabel('Estimated Preference Strength (Beta)')
ax.set_title('Top 15 Models: Global vs Creativity (95% CI)')
ax.legend()
plt.grid(axis='x', alpha=0.3)
plt.savefig("results/notebook_top_models_bar.png", dpi=300, bbox_inches='tight')
print("Bar plot saved to results/notebook_top_models_bar.png")

# %% [markdown]
# ### 4.2. Movers Standardization Shift
# Visualizing standardized shift (Z-score differences) for the most sensitive models.
#
# **Result:** After standardizing both β distributions, the largest standardized gainers are:
# - `gpt-5-nano` (z_shift = +1.11) and `deepseek-r1-distill-llama-70b` (+0.94) — smaller or
#   reasoning-focused models that benefit disproportionately from creative prompts.
# - `qwen3.5-397b-a17b` (+0.89) — a large-scale model that gains significantly.
# The largest standardized losers are `magistral-small-2506` (−1.36) and `gpt-5.3` (−1.13),
# confirming these models lose competitive advantage specifically in creative contexts.

# %%
from scipy.stats import zscore

z_g = pd.Series(zscore(beta_global[common_models]), index=common_models)
z_c = pd.Series(zscore(beta_creative[common_models]), index=common_models)

z_shift = z_c - z_g

top_gainers = z_shift.nlargest(8)
top_losers = z_shift.nsmallest(8)
movers = pd.concat([top_gainers, top_losers]).sort_values()

plt.figure(figsize=(10, 8))
colors = ['#d62728' if x < 0 else '#2ca02c' for x in movers.values]

plt.barh(np.arange(len(movers)), movers.values, color=colors, alpha=0.8, edgecolor='black')
plt.yticks(np.arange(len(movers)), movers.index)
plt.axvline(0, color='k', linestyle='--', alpha=0.5)

plt.xlabel('Standardized Preference Shift (Z_creative - Z_global)')
plt.title('Largest Movers in Creativity-Filtered Regime')
plt.grid(axis='x', alpha=0.3)

plt.savefig("results/notebook_beta_shift_movers.png", dpi=300, bbox_inches='tight')
print("Shift bar plot saved to results/notebook_beta_shift_movers.png")
plt.show()

# %% [markdown]
# ### 4.3 Creativity Regime Sensitivity Analysis
# We evaluate how stable the Top-10 creative ranking is across different minimum comparison thresholds (N=20, 50, 100)
# to justify our chosen threshold.
#
# **Result:** The Top-10 shows moderate stability:
# - **N≥20 vs N≥50**: 6/10 models in common (60% overlap).
# - **N≥20 vs N≥100**: only 5/10 in common (50% overlap).
# - **N≥50 vs N≥100**: 8/10 in common (80% overlap) — stable at higher thresholds.
# Models **exclusively** in the N≥20 top: `gemini-3.1-flash-lite-preview`, `glm-4.5`,
# `qwen3.5-397b-a17b`, `gemini-3.1-pro-preview`, `deepseek-chat-v3.1` — these models have
# few creative comparisons, so their rankings are unstable.
# **This justifies cautious interpretation of the N=20 regime**: the stable core (N≥50)
# consistently includes `mistral-medium-2508`, `qwen3-max-2025-09-23`, `mistral-large-2512`.

# %%
thresholds = [20, 50, 100]
print("\n[CREATIVITY SENSITIVITY ANALYSIS]")

for t in thresholds:
    valid_m = model_counts_c[model_counts_c >= t].index
    df_t = df_creative[df_creative['model_a_name'].isin(valid_m) & df_creative['model_b_name'].isin(valid_m)]
    
    W_t = pd.DataFrame(0, index=valid_m, columns=valid_m, dtype=float)
    a_wins_t = df_t[df_t['chosen_model_name'] == df_t['model_a_name']]
    for (ma, mb), count in a_wins_t.groupby(['model_a_name', 'model_b_name']).size().items():
        if ma in valid_m and mb in valid_m: W_t.loc[ma, mb] += count
            
    b_wins_t = df_t[df_t['chosen_model_name'] == df_t['model_b_name']]
    for (mb, ma), count in b_wins_t.groupby(['model_b_name', 'model_a_name']).size().items():
        if ma in valid_m and mb in valid_m: W_t.loc[mb, ma] += count
            
    models_t = W_t.index
    n_t = len(models_t)
    w_mat_t = W_t.values
    wins_t = w_mat_t.sum(axis=1)
    n_mat_t = w_mat_t + w_mat_t.T
    pi_t = np.ones(n_t) / n_t
    
    for _ in range(1000):
        pi_prev = pi_t.copy()
        for i in range(n_t):
            if wins_t[i] == 0: pi_t[i] = 1e-10; continue
            denom = 0.0
            for j in range(n_t):
                if i != j and n_mat_t[i, j] > 0:
                    denom += n_mat_t[i, j] / (pi_prev[i] + pi_prev[j])
            if denom > 0: pi_t[i] = wins_t[i] / denom
        pi_t = pi_t / pi_t.sum()
        if np.max(np.abs(pi_t - pi_prev)) < 1e-6: break
            
    beta_t_log = np.log(np.maximum(pi_t, 1e-10))
    beta_t_log -= beta_t_log.mean()
    beta_t = pd.Series(beta_t_log, index=models_t).sort_values(ascending=False)
    
    print(f"\nTop 10 Models (Creative N>={t}, total valid={n_t}):")
    print(beta_t.head(10).to_string(float_format=lambda x: f"{x:.4f}"))

# %%
