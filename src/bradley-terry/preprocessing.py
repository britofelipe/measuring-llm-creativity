import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Tuple, List

def load_data(dataset_name: str = "ministere-culture/comparia-votes") -> pd.DataFrame:
    """Loads the dataset from HuggingFace."""
    print(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    df = dataset.to_pandas()
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes invalid rows and normalizes model names."""
    df = df.dropna(subset=['model_a_name', 'model_b_name'])
    # Valid if it's a tie, OR if there is a chosen model
    valid_mask = (df['both_equal'] == True) | (df['chosen_model_name'].notna())
    df = df[valid_mask].copy()
    
    df.loc[:, 'model_a_name'] = df['model_a_name'].astype(str).str.lower().str.strip()
    df.loc[:, 'model_b_name'] = df['model_b_name'].astype(str).str.lower().str.strip()
    
    notna_mask = df['chosen_model_name'].notna()
    df.loc[notna_mask, 'chosen_model_name'] = df.loc[notna_mask, 'chosen_model_name'].astype(str).str.lower().str.strip()
    return df

def justify_and_filter_N(df: pd.DataFrame, min_comparisons: int) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Filters the dataset to only include models that appear in at least `min_comparisons` valid matchups.
    Also plots/prints the distribution of comparison counts for justification.
    """
    model_counts = pd.concat([df['model_a_name'], df['model_b_name']]).value_counts()
    
    print("\n--- Model Comparison Distribution ---")
    print(f"Total Unique Models: {len(model_counts)}")
    print(f"Mean comparisons per model: {model_counts.mean():.1f}")
    print(f"Median comparisons per model: {model_counts.median():.1f}")
    print(f"Max comparisons: {model_counts.max()}")
    print("Quantiles:")
    print(model_counts.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
    
    valid_models = model_counts[model_counts >= min_comparisons].index
    print(f"Filtering to {len(valid_models)} models with >= {min_comparisons} comparisons.")
    
    filtered_df = df[df['model_a_name'].isin(valid_models) & df['model_b_name'].isin(valid_models)].copy()
    return filtered_df, valid_models

def build_wins_matrix(df: pd.DataFrame, models: pd.Index) -> pd.DataFrame:
    """
    Builds the W[i,j] gain matrix where W[i,j] is the number of times model i was preferred to model j.
    """
    W = pd.DataFrame(0, index=models, columns=models)
    
    # Count times Model A was chosen
    a_wins = df[df['chosen_model_name'] == df['model_a_name']]
    a_counts = a_wins.groupby(['model_a_name', 'model_b_name']).size()
    for (ma, mb), count in a_counts.items():
        if ma in models and mb in models:
            W.loc[ma, mb] += count
            
    # Count times Model B was chosen
    b_wins = df[df['chosen_model_name'] == df['model_b_name']]
    b_counts = b_wins.groupby(['model_b_name', 'model_a_name']).size()
    for (mb, ma), count in b_counts.items():
        if ma in models and mb in models:
            W.loc[mb, ma] += count
            
    return W

def build_ties_matrix(df: pd.DataFrame, models: pd.Index) -> pd.DataFrame:
    """
    Builds the T[i,j] matrix counting the number of ties (both_equal == True) between model i and model j.
    """
    T = pd.DataFrame(0, index=models, columns=models)
    ties_df = df[df['both_equal'] == True]
    
    tie_counts = ties_df.groupby(['model_a_name', 'model_b_name']).size()
    for (ma, mb), count in tie_counts.items():
        if ma in models and mb in models:
            T.loc[ma, mb] += count
            T.loc[mb, ma] += count  # Symmetric
            
    return T

def get_base_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filters only votes where both_equal == False for the base Bradley-Terry model."""
    return df[df['both_equal'] == False].copy()

def get_creative_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filters conversations where at least one of the conv_creative_* is True."""
    c_a = df['conv_creative_a'].fillna(False).astype(bool)
    c_b = df['conv_creative_b'].fillna(False).astype(bool)
    return df[c_a | c_b].copy()

if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    
    print("\n[GLOBAL RANKING REGIME]")
    df_base = get_base_data(df_clean)
    # We choose N=100 initially to see how many models are retained
    df_filtered, valid_models = justify_and_filter_N(df_base, min_comparisons=100)
    W_base = build_wins_matrix(df_filtered, valid_models)
    print(f"Base matrix shape: {W_base.shape}")
    
    print("\n[CREATIVITY RANKING REGIME]")
    df_creative = get_creative_data(df_clean)
    df_creative_filtered, valid_models_c = justify_and_filter_N(df_creative, min_comparisons=20)
    W_creative = build_wins_matrix(df_creative_filtered, valid_models_c)
    print(f"Creative matrix shape: {W_creative.shape}")
