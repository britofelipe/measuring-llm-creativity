import pandas as pd
import numpy as np
import statsmodels.api as sm
from preprocessing import load_data, clean_data

def get_text_length(conv) -> int:
    """Extracts approximate token/character length from conversation dictionary."""
    if not isinstance(conv, np.ndarray) and not isinstance(conv, list):
        return 0
    total_len = 0
    for message in conv:
        if isinstance(message, dict) and 'content' in message:
            total_len += len(str(message.get('content', '')))
    return total_len

def build_covariate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Filter valid votes without ties
    df = df[df['both_equal'] == False].copy()
    
    # Target: 1 if model_a won, 0 if model_b won
    df['y_a_wins'] = (df['chosen_model_name'] == df['model_a_name']).astype(int)
    
    print("Extracting output lengths from conversations...")
    df['len_a'] = df['conversation_a'].apply(get_text_length)
    df['len_b'] = df['conversation_b'].apply(get_text_length)
    df['len_diff'] = df['len_a'] - df['len_b']
    
    df['conv_turns'] = pd.to_numeric(df['conv_turns'], errors='coerce').fillna(1)
    df['category'] = df['selected_category'].fillna('Unknown').astype(str)
    
    return df

def fit_covariate_model(df_cov: pd.DataFrame, top_n: int = 30):
    model_counts = pd.concat([df_cov['model_a_name'], df_cov['model_b_name']]).value_counts()
    top_models = model_counts.nlargest(top_n).index.tolist()
    
    df_sub = df_cov[df_cov['model_a_name'].isin(top_models) & df_cov['model_b_name'].isin(top_models)].copy()
    
    print(f"\nFitting on {len(df_sub)} valid matchups for Top {top_n} models...")
    
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
    
    # Covariates
    X = X_models.copy()
    X['len_diff_scaled'] = df_sub['len_diff'] / 1000.0  # Thousands of characters
    X['conv_turns'] = df_sub['conv_turns']
    
    cat_dummies = pd.get_dummies(df_sub['category'], prefix='cat', drop_first=True, dtype=float)
    X = pd.concat([X, cat_dummies], axis=1)
    
    X = sm.add_constant(X)
    y = df_sub['y_a_wins']
    
    print("\nFitting Logistic Regression (Covariate Extended BT)...")
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    
    print("\n--- Covariate Model Summary (Excerpt) ---")
    # Only show coefficients of covariates, not all 30 models
    covariate_cols = ['const', 'len_diff_scaled', 'conv_turns'] + cat_dummies.columns.tolist()
    print(result.summary().tables[1].data[0]) # header
    for row in result.summary().tables[1].data[1:]:
        if row[0].strip() in covariate_cols:
            print(row)
    
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

    import matplotlib.pyplot as plt
    import os
    os.makedirs("results", exist_ok=True)
    
    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues
    cov_names = [c for c in params.index if 'len_diff' in c or 'conv_turns' in c or 'cat_' in c]
    
    if cov_names:
        # Create summary table
        summary_df = pd.DataFrame({
            'Estimate (Log-Odds)': params[cov_names],
            'CI Lower': conf[0][cov_names],
            'CI Upper': conf[1][cov_names],
            'P-value': pvals[cov_names]
        }).sort_values('Estimate (Log-Odds)')
        
        summary_df.to_csv("results/covariate_summary.csv")
        
        vals = summary_df['Estimate (Log-Odds)']
        errors = np.abs(summary_df['CI Upper'] - vals).values
        
        # Color: Green if positive sig, Red if negative sig, Gray if not sig
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
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ca02c', label='Sig. Positive'),
                           Patch(facecolor='#d62728', label='Sig. Negative'),
                           Patch(facecolor='gray', label='Not Significant (p >= 0.05)')]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.savefig("results/covariate_effects.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("\nCovariate effects plot saved to results/covariate_effects.png")
        print("Model summary table saved to results/covariate_summary.csv")

if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    
    df_cov = build_covariate_dataset(df_clean)
    
    print("\nSample Length Calculations:")
    print(df_cov[['model_a_name', 'len_a', 'model_b_name', 'len_b', 'len_diff']].head())
    
    fit_covariate_model(df_cov, top_n=30)
