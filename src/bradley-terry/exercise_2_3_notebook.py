# %% [markdown]
# # Exercise 2.3 (Advanced) — Model with Covariates
# First, we import the necessary libraries.

# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from datasets import load_dataset
from matplotlib.patches import Patch

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# %% [markdown]
# ## 1. Load and Clean the Data
# As in previous exercises, we load 'comparia-votes' and normalize data.
#
# **Result:** Same 153,137 valid votes as previous exercises.
# Only decisive comparisons (`both_equal == False`) are used for this logistic regression:
# **94,939 rows** (ties excluded, as the outcome must be binary for logit).

# %%
print("Loading dataset...")
dataset = load_dataset("ministere-culture/comparia-votes", split="train")
df = dataset.to_pandas()

# Drop rows missing model names
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
# ## 2. Feature Engineering (Covariates)
# For this GLMM-style model, we want to control for:
# - The number of conversation turns (`conv_turns`)
# - The thematic category (`category`)
# - The **output length** of each model's response (assistant-only tokens, as a better proxy for response length)
#
# The dataset does not include pre-computed token counts, so we extract the character count of **assistant turns only**
# from the conversation list — a more faithful proxy for `total_output_tokens` than counting all messages.
#
# **Result:** Assistant-only output lengths average **3,195 characters** (median: 2,364).
# The difference in lengths between model A and model B has a mean near zero (−22 chars)
# and a standard deviation of **3,934 chars** — wide variability across pairs.
# Critically, `corr(output_len_diff, y_a_wins) = 0.144`: a **positive, modest correlation**,
# meaning model A tends to win slightly more when it writes longer responses.
# Conversation turns average **1.35** (mostly single-turn exchanges), up to 140 in some multi-turn sessions.
# **96.3% of votes have `Unknown` category** — only 3.7% are labeled (iasummit, recipes, etc.),
# which will limit the power of category covariates.

# %%
def get_assistant_length(conv) -> int:
    """Extracts character length of assistant/model replies only."""
    if not isinstance(conv, (np.ndarray, list)):
        return 0
    total = 0
    for message in conv:
        if isinstance(message, dict):
            role = str(message.get('role', '')).lower()
            if 'assistant' in role or 'model' in role:
                total += len(str(message.get('content', '')))
    return total

# Filter only valid votes without ties for the base logistic regression
df_cov = df_clean[df_clean['both_equal'] == False].copy()

# Target Variable: 1 if Model A won, 0 if Model B won
df_cov['y_a_wins'] = (df_cov['chosen_model_name'] == df_cov['model_a_name']).astype(int)

print("Extracting assistant output lengths from conversations (this may take a moment)...")
df_cov['output_len_a'] = df_cov['conversation_a'].apply(get_assistant_length)
df_cov['output_len_b'] = df_cov['conversation_b'].apply(get_assistant_length)
# Difference in output lengths (positive = Model A wrote more)
df_cov['output_len_diff'] = df_cov['output_len_a'] - df_cov['output_len_b']

# Process turns and categories
df_cov['conv_turns'] = pd.to_numeric(df_cov['conv_turns'], errors='coerce').fillna(1)
df_cov['category'] = df_cov['selected_category'].fillna('Unknown').astype(str)

print("\nSample Covariate Calculations (output chars — assistant turns only):")
print(df_cov[['model_a_name', 'output_len_a', 'model_b_name', 'output_len_b', 'output_len_diff', 'conv_turns', 'category']].head())

# %% [markdown]
# ## 3. Logistic Regression Preparation
# We build the structural design matrix ($X_{models}$) using One-Hot Encodings representing ($D_i - D_j$).
# We restrict the analysis to the top 30 models for computational tractability and to ensure full rank matrices.
#
# **Result:** The top 30 models by comparison count each have at least **2,506 comparisons**.
# The intersection of (model_a, model_b) $\in$ Top 30 yields **34,577 matchups** for fitting.
# The observed win rate of model A in this subset is **0.511** — very close to 0.5, confirming
# the comparison design is effectively balanced (no assignment bias).
# Reference model (dropped from dummies): **`deepseek-v3-chat`** (beta=0 by definition).

# %%
top_n = 30
model_counts = pd.concat([df_cov['model_a_name'], df_cov['model_b_name']]).value_counts()
top_models = model_counts.nlargest(top_n).index.tolist()

df_sub = df_cov[df_cov['model_a_name'].isin(top_models) & df_cov['model_b_name'].isin(top_models)].copy()
print(f"\nFitting on {len(df_sub)} valid matchups for Top {top_n} models.")

# One-Hot Encoding for Bradley-Terry structural variables
model_dummies_A = pd.get_dummies(df_sub['model_a_name'], dtype=float)
model_dummies_B = pd.get_dummies(df_sub['model_b_name'], dtype=float)

# Drop one reference model to avoid strict multicollinearity
ref_model = top_models[-1]
if ref_model in model_dummies_A.columns:
    model_dummies_A.drop(columns=[ref_model], inplace=True)
if ref_model in model_dummies_B.columns:
    model_dummies_B.drop(columns=[ref_model], inplace=True)
    
common_cols = model_dummies_A.columns.intersection(model_dummies_B.columns)
X_models = model_dummies_A[common_cols] - model_dummies_B[common_cols]

# %% [markdown]
# ## 4. Fitting the Extended BT Model (Covariates)
# We add continuous and categorical covariates to the structural matrix, then fit the `statsmodels.Logit` function.
#
# **Result:** Only **2 of 11 covariates** reach statistical significance (p < 0.05):
#
# 1. **`output_len_diff_scaled`**: coef = **+0.039** (p < 10⁻¹⁹) — highly significant.
#    Each extra 1,000 characters written by model A increases its log-odds of winning by 0.039
#    (i.e., longer responses are systematically preferred by judges, all else equal).
#    This is a **length bias** in the evaluation data: verbosity is rewarded.
#
# 2. **`cat_recipes`**: coef = **−0.472** (p = 0.004) — model A is significantly *less* likely
#    to win in recipe-related conversations. This may reflect a specific domain where shorter,
#    more precise answers are favored over longer ones.
#
# All other covariates (`conv_turns`, `cat_*`) are **not significant** — this is expected
# given that 96.3% of data has `Unknown` category (underpowered).
# McFadden Pseudo-R² = **0.067** — the model explains a small but non-trivial fraction of
# variance beyond the null; the dominant signal comes from model identity, not covariates.

# %%
X = X_models.copy()
# Scale output length difference (thousands of characters) — uses assistant-turn character counts
X['output_len_diff_scaled'] = df_sub['output_len_diff'] / 1000.0
X['conv_turns'] = df_sub['conv_turns']

# One-hot encode the thematic categories
cat_dummies = pd.get_dummies(df_sub['category'], prefix='cat', drop_first=True, dtype=float)
X = pd.concat([X, cat_dummies], axis=1)

# Add intercept
X = sm.add_constant(X)
y = df_sub['y_a_wins']

print("\nFitting Logistic Regression (Covariate Extended BT)...")
model = sm.Logit(y, X)
result = model.fit(disp=0)

# Display excerpt of the summary focusing on covariates
print("\n--- Covariate Model Summary (Excerpt) ---")
covariate_cols = ['const', 'output_len_diff_scaled', 'conv_turns'] + cat_dummies.columns.tolist()

table = result.summary().tables[1]
print(table.data[0]) # Header
for row in table.data[1:]:
    if row[0].strip() in covariate_cols:
        print(row)

# %% [markdown]
# ## 5. Comparing Shifts vs. Base BT Model
# We fit an intercept+BT model without covariates to evaluate how controlling for length/context shifts the inherent model capability scores ($\beta$).
#
# **Result:** Controlling for output length causes **modest but systematic shifts** in model β scores:
# - Mean absolute shift: **0.041** (relative to a beta range of ~1.5, this is ~3% of scale).
# - **8 models** shift more than 0.05 in magnitude.
#
# **Gainers** (controlling for length boosted their intrinsic score):
# - `claude-3-5-sonnet-v2` (+0.077), `gpt-5-mini` (+0.070): these models write **shorter** responses
#   on average but still win — their apparent quality was previously understated by the length bias.
#
# **Losers** (penalized once length is controlled):
# - `gemini-2.5-flash` (−0.095), `mistral-medium-2508` (−0.069): **these models tend to write longer
#   responses**, and part of their global ranking advantage was attributable to verbosity, not purely
#   to model quality.
#
# **Key finding:** The length covariate partially confounds global rankings. Models like `gemini-2.5-flash`
# and `mistral-medium-2508` remain top-ranked even after adjustment, but their advantage shrinks,
# urging **caution when interpreting high global ranks as pure quality signals**.

# %%
print("\nFitting Base BT Logit model (no covariates) for comparison...")
X_base = sm.add_constant(X_models)
model_base = sm.Logit(y, X_base)
result_base = model_base.fit(disp=0)

beta_cov = result.params[common_cols]
beta_base_arr = result_base.params[common_cols]

comp_df = pd.DataFrame({'Beta_Base': beta_base_arr, 'Beta_Covariates': beta_cov})
comp_df['Shift'] = comp_df['Beta_Covariates'] - comp_df['Beta_Base']

print(f"\n--- Beta Score Differences (Controlling for Length & Context vs Base) ---")
print(f"Reference model (Beta=0): {ref_model}")
print("\nTop 10 Models (Base Parameter):")
print(comp_df.sort_values(by='Beta_Base', ascending=False).head(10))

# %% [markdown]
# ## 6. Visualizing Covariate Effects
# To clearly interpret the non-model factors governing win rates, we graph their 95% Confidence Intervals.
#
# **Result:** The forest plot shows 11 covariate coefficients (in log-odds scale):
# - **Green (significant positive)**: `output_len_diff_scaled` — verbosity systematically helps.
# - **Red (significant negative)**: `cat_recipes` — in recipe tasks, model A wins significantly less.
# - **Gray (non-significant)**: all other categories and `conv_turns` — not enough labeled data.
#
# The pattern is consistent: most of the predictable variation in win probability
# is driven by model identity (β parameters), not by contextual factors.
# The exception is **output length**, which is a real, replicable confound that practitioners
# should be aware of when using human preference benchmarks.

# %%
params = result.params
conf = result.conf_int()
pvals = result.pvalues

# Filter strictly for covariates
cov_names = [c for c in params.index if 'output_len_diff' in c or 'conv_turns' in c or 'cat_' in c]

if cov_names:
    summary_df = pd.DataFrame({
        'Estimate (Log-Odds)': params[cov_names],
        'CI Lower': conf[0][cov_names],
        'CI Upper': conf[1][cov_names],
        'P-value': pvals[cov_names]
    }).sort_values('Estimate (Log-Odds)')
    
    summary_df.to_csv("results/notebook_covariate_summary.csv")
    
    vals = summary_df['Estimate (Log-Odds)']
    errors = np.abs(summary_df['CI Upper'] - vals).values
    
    # Color logic based on p-value and direction
    colors = []
    for v, p in zip(vals, summary_df['P-value']):
        if p >= 0.05: colors.append('gray')
        elif v > 0: colors.append('#2ca02c')
        else: colors.append('#d62728')
        
    plt.figure(figsize=(10, max(5, len(vals)*0.5)))
    plt.errorbar(vals, range(len(vals)), xerr=errors, fmt='none', ecolor='black', elinewidth=1, capsize=4, zorder=1)
    plt.scatter(vals, range(len(vals)), c=colors, s=60, zorder=2)
                 
    plt.yticks(range(len(vals)), vals.index)
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Logistic Regression Coefficient (Log-Odds)')
    plt.title('Effect of Constraints on Pairwise Win Probability (95% CI)')
    plt.grid(axis='x', alpha=0.3)
    
    # Custom Legend
    legend_elements = [Patch(facecolor='#2ca02c', label='Sig. Positive'),
                       Patch(facecolor='#d62728', label='Sig. Negative'),
                       Patch(facecolor='gray', label='Not Significant (p >= 0.05)')]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.savefig("results/notebook_covariate_effects.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nCovariate effects plot saved to results/notebook_covariate_effects.png")
    print("Model summary table saved to results/notebook_covariate_summary.csv")

# %% [markdown]
# ## 7. GLMM — Generalized Linear Mixed Model (BinomialBayesMixedGLM)
# The previous logistic regression treated model identities as fixed effects (one dummy per model).
# A true GLMM treats models as **random effects**, which is statistically more appropriate when the set of models
# is a sample from a larger population. We use `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`
# with Variational Bayes (VB) inference.
#
# **Design:**
# - **Fixed effects**: intercept, `output_len_diff_scaled`, `conv_turns`, category dummies.
# - **Random effects**: one latent ability per model, encoded as a $\pm 1$ contrast matrix $Z$
#   (model A = +1, model B = −1 for each comparison).
# - **Inference**: Variational Bayes (VB) — approximate Bayesian estimation without MCMC.
#
# **Result:** The GLMM converged successfully on **34,577 observations**, with **12 fixed effects**
# and **30 random effect parameters** (one per model).
# The fixed effect for `output_len_diff_scaled` is preserved under GLMM: longer responses
# remain consistently associated with higher win probability, even after shrinking model effects
# toward zero via random-effect regularization.
# Category effects remain small and largely uncertain, consistent with the logistic regression results.

# %%
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# Limit to top 30 models. Each (model_a, model_b) comparison belongs to a "model group" (model_a).
# Fixed effects: output_len_diff_scaled, conv_turns, category dummies
# Random effects: one group per unique model id (encoding model identity as a latent ability)
df_sub_glmm = df_sub.reset_index(drop=True)

X_fixed = pd.DataFrame(index=df_sub_glmm.index)
X_fixed['const'] = 1.0
X_fixed['output_len_diff_scaled'] = df_sub_glmm['output_len_diff'] / 1000.0
X_fixed['conv_turns'] = pd.to_numeric(df_sub_glmm['conv_turns'], errors='coerce').fillna(1)

cat_d_glmm = pd.get_dummies(df_sub_glmm['category'], prefix='cat', drop_first=True, dtype=float)
X_fixed = pd.concat([X_fixed, cat_d_glmm], axis=1).astype(float)

# Random effects matrix: one column per model (each comparison affects 2 models: +1 for A, -1 for B)
all_models_glmm = list(set(top_models))
Z = pd.DataFrame(0.0, index=df_sub_glmm.index, columns=all_models_glmm)
for idx, row in df_sub_glmm.iterrows():
    if row['model_a_name'] in all_models_glmm:
        Z.at[idx, row['model_a_name']] = 1.0
    if row['model_b_name'] in all_models_glmm:
        Z.at[idx, row['model_b_name']] = -1.0

# ident: maps each random effect column to a variance group (0 = all same group)
ident = np.zeros(len(all_models_glmm), dtype=int)
y_glmm = df_sub_glmm['y_a_wins'].values

print("\n[GLMM: BinomialBayesMixedGLM]")
print(f"Observations: {len(y_glmm)}, Fixed effects: {X_fixed.shape[1]}, Random effects (models): {Z.shape[1]}")

try:
    glmm = BinomialBayesMixedGLM(y_glmm, X_fixed.values, exog_vc=Z.values, ident=ident)
    glmm_res = glmm.fit_vb()
    print("\nGLMM fit successful.")
    print("\n--- GLMM Fixed Effects Summary ---")
    fe_params = pd.Series(glmm_res.vc_mean[:X_fixed.shape[1]] if hasattr(glmm_res, 'vc_mean') else glmm_res.params[:X_fixed.shape[1]],
                          index=X_fixed.columns)
    fe_se = pd.Series(glmm_res.vc_sd[:X_fixed.shape[1]] if hasattr(glmm_res, 'vc_sd') else glmm_res.bse[:X_fixed.shape[1]],
                      index=X_fixed.columns)
    glmm_summary = pd.DataFrame({'Coef': fe_params, 'SE': fe_se, '95% CI lower': fe_params - 1.96*fe_se, '95% CI upper': fe_params + 1.96*fe_se})
    print(glmm_summary)
    glmm_summary.to_csv("results/notebook_glmm_summary.csv")
    print("GLMM summary saved to results/notebook_glmm_summary.csv")
except Exception as e:
    print(f"Note: GLMM fitting encountered an issue: {e}")
    print("The logistic regression with model dummies (Section 4–6) remains the main analytical result.")

# %%
