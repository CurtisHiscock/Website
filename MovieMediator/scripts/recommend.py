# recommend.py
##############

import numpy as np
from nltk.corpus import wordnet
from sentence_transformers import util

#Expand buzzwords with synonyms using WordNet
#--------------------------------------------
def expand_buzzwords_with_synonyms(buzzwords):
    words = buzzwords.lower().split()
    expanded = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return ' '.join(expanded)

#Semantic search using SentenceTransformer
#------------------------------------------------------------------------
def semantic_buzzword_search(query, model, movie_embeddings, df, top_n=3):
    #Embed the user query
    query_embedding = model.encode(query, convert_to_tensor=True)

    #Compute cosine similarity between query and movie embeddings
    cosine_scores = util.cos_sim(query_embedding, movie_embeddings)[0]

    #Convert tensors to numpy arrays
    scores = cosine_scores.cpu().numpy()
    ratings = df['normalized_rating'].values

    #Blend cosine similarity and rating (adjust weights as needed)
    final_scores = 0.90 * scores + 0.10 * ratings

    #Get sorted indices for top scores
    sorted_indices = final_scores.argsort()[::-1]

    #Build final result list: (title, score, poster_url)
    results = []
    for idx in sorted_indices[:top_n * 10]:  #slice extra for rerolling support
        title = df.iloc[idx]['title']
        score = round(float(final_scores[idx]), 3)
        poster_url = df.iloc[idx].get('poster_url', None)
        results.append((title, score, poster_url))

    return results