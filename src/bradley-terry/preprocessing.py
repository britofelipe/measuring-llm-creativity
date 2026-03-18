import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
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

def plot_comparison_distribution(counts: pd.Series, regime_name: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    counts.hist(bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of Comparisons per Model ({regime_name})')
    plt.xlabel('Number of Comparisons (N)')
    plt.ylabel('Number of Models (Log Scale)')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {output_path}")

def plot_threshold_sensitivity(counts: pd.Series, regime_name: str, output_path: str):
    thresholds = [10, 20, 50, 100, 200, 500]
    retained = [sum(counts >= t) for t in thresholds]
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, retained, marker='o', linestyle='-', color='purple', lw=2)
    plt.title(f'Threshold Sensitivity: Models Retained ({regime_name})')
    plt.xlabel('Minimum Comparisons Threshold (N)')
    plt.ylabel('Number of Models Retained')
    plt.grid(True, alpha=0.3)
    for i, txt in enumerate(retained):
        plt.annotate(str(txt), (thresholds[i], retained[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved threshold sensitivity plot to {output_path}")

def justify_and_filter_N(df: pd.DataFrame, min_comparisons: int, regime_name: str = "Global") -> Tuple[pd.DataFrame, pd.Index]:
    model_counts = pd.concat([df['model_a_name'], df['model_b_name']]).value_counts()
    
    print(f"\n--- Model Comparison Distribution ({regime_name}) ---")
    print(f"Total Unique Models: {len(model_counts)}")
    print(f"Mean comparisons per model: {model_counts.mean():.1f}")
    print(f"Median comparisons per model: {model_counts.median():.1f}")
    
    plot_comparison_distribution(model_counts, regime_name, f"results/dist_{regime_name.lower()}_comparisons.png")
    plot_threshold_sensitivity(model_counts, regime_name, f"results/sens_{regime_name.lower()}_thresholds.png")
    
    valid_models = model_counts[model_counts >= min_comparisons].index
    print(f"Filtering to {len(valid_models)} models with >= {min_comparisons} comparisons.")
    
    filtered_df = df[df['model_a_name'].isin(valid_models) & df['model_b_name'].isin(valid_models)].copy()
    return filtered_df, valid_models

def verify_graph_connectivity(W: pd.DataFrame) -> pd.DataFrame:
    adj = (W.values + W.values.T) > 0
    graph = csr_matrix(adj)
    
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    if n_components == 1:
        print("Graph connectivity verified: The matrix subgraph forms a single strong component.")
        return W
        
    print(f"WARNING: Graph disconnected into {n_components} components. BT parameters are not jointly identifiable.")
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_component_label = unique_labels[np.argmax(counts)]
    
    keep_idx = np.where(labels == largest_component_label)[0]
    models_to_keep = W.index[keep_idx]
    
    print(f"Restricting to largest component: keeping {len(models_to_keep)} out of {len(W)} models.")
    return W.loc[models_to_keep, models_to_keep]

def build_wins_matrix(df: pd.DataFrame, models: pd.Index) -> pd.DataFrame:
    W = pd.DataFrame(0, index=models, columns=models, dtype=float)
    
    a_wins = df[df['chosen_model_name'] == df['model_a_name']]
    a_counts = a_wins.groupby(['model_a_name', 'model_b_name']).size()
    for (ma, mb), count in a_counts.items():
        if ma in models and mb in models:
            W.loc[ma, mb] += count
            
    b_wins = df[df['chosen_model_name'] == df['model_b_name']]
    b_counts = b_wins.groupby(['model_b_name', 'model_a_name']).size()
    for (mb, ma), count in b_counts.items():
        if ma in models and mb in models:
            W.loc[mb, ma] += count
            
    W = verify_graph_connectivity(W)
    return W

def build_ties_matrix(df: pd.DataFrame, models: pd.Index) -> pd.DataFrame:
    T = pd.DataFrame(0, index=models, columns=models, dtype=float)
    ties_df = df[df['both_equal'] == True]
    
    tie_counts = ties_df.groupby(['model_a_name', 'model_b_name']).size()
    for (ma, mb), count in tie_counts.items():
        if ma in models and mb in models:
            T.loc[ma, mb] += count
            T.loc[mb, ma] += count
            
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
    df_filtered, valid_models = justify_and_filter_N(df_base, min_comparisons=100, regime_name="Global")
    W_base = build_wins_matrix(df_filtered, valid_models)
    print(f"Base matrix shape: {W_base.shape}")
    
    print("\n[CREATIVITY RANKING REGIME]")
    df_creative = get_creative_data(df_clean)
    df_creative_filtered, valid_models_c = justify_and_filter_N(df_creative, min_comparisons=20, regime_name="Creative")
    W_creative = build_wins_matrix(df_creative_filtered, valid_models_c)
    print(f"Creative matrix shape: {W_creative.shape}")
