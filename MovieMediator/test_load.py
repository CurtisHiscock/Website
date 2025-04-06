from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import os

try:
    print("Testing model load...")
    model_path = os.path.join(os.path.dirname(__file__), "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_path)
    print("✅ Model loaded")

    print("Testing data load...")
    df_path = "/home/curtavtz/public_html/TheMovieMediator/preprocessed_df.pkl"
    embeddings_path = "/home/curtavtz/public_html/TheMovieMediator/embeddings.pt"
    
    df = pd.read_pickle(df_path)
    embeddings = torch.load(embeddings_path)
    print(f"✅ Data loaded: {len(df)} movies, embeddings shape: {embeddings.shape}")

except Exception as e:
    print("❌ FAILED:", e)
    import traceback
    traceback.print_exc()
