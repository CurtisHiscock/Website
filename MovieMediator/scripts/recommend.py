# recommend.py
##############

import numpy as np
import torch
from nltk.corpus import wordnet
import torch.nn.functional as F  # <== for cosine similarity

# Expand buzzwords with synonyms using WordNet
def expand_buzzwords_with_synonyms(buzzwords):
    words = buzzwords.lower().split()
    expanded = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return ' '.join(expanded)

# Semantic search using SimpleEmbedder + PyTorch cosine similarity
def semantic_buzzword_search(query, model, movie_embeddings, df, top_n=3):
    # Expand query with synonyms
    expanded_query = expand_buzzwords_with_synonyms(query)

    # Embed the query using your custom embedder
    query_embedding = model.encode(expanded_query)  # shape: [1, embedding_dim]

    # Compute cosine similarity between query and all movie embeddings
    cosine_scores = F.cosine_similarity(query_embedding, movie_embeddings)  # shape: [num_movies]

    # Convert to numpy
    scores = cosine_scores.cpu().numpy()
    ratings = df['normalized_rating'].values

    # Blend scores with ratings
    final_scores = 0.90 * scores + 0.10 * ratings

    # Sort by final blended score
    sorted_indices = final_scores.argsort()[::-1]

    # Build results
    results = []
    for idx in sorted_indices[:top_n * 10]:  # extra for re-roll support
        title = df.iloc[idx]['title']
        score = round(float(final_scores[idx]), 3)
        poster_url = df.iloc[idx].get('poster_url', None)
        results.append((title, score, poster_url))

    return results
