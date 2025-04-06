# data_loader.py
################

import os
import pandas as pd
import torch

#Load cached data and embeddings
#--------------------------------------------------------------------
def load_cached_data():
    """
    Loads preprocessed movie DataFrame and semantic embeddings from disk.
    Returns:
        df (DataFrame): Preprocessed movie dataset
        embeddings (Tensor): Semantic movie embeddings
    """

    #File paths
    base_dir = os.path.dirname(__file__)
    df_path = os.path.join(os.path.dirname(__file__), "../preprocessed_df.pkl")
    embeddings_path = os.path.join(os.path.dirname(__file__), "../embeddings.pt")

    #Check if both files exist (required for running)
    if not os.path.exists(df_path) or not os.path.exists(embeddings_path):
        raise FileNotFoundError("Cached data not found. Run the preprocessing script first.")

    #Load preprocessed movie dataset
    df = pd.read_pickle(df_path)

    #Load semantic vector embeddings (Tensor)
    embeddings = torch.load(embeddings_path)

    return df, embeddings