import pandas as pd
import numpy as np
import json
import os
import nltk
import torch
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

#print(os.getcwd()) #Where the code is running from

#API key for getting posters
TMDB_API_KEY = "73b4f8a56d4c5d4442c8b569a755c3b6"

#Global helper functions
########################

#Function that gets the directors name, NaN if not listed
#--------------------------------------------------------
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

#Function that returns top 3 elements or full list depending on which is larger
#----------------------------------------------------------------
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

#Function to convert all strings to lower case and strip names of spaces
#------------------------------------------------------------------------
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        return ''

#Metadata soup
#-------------
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

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
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, movie_embeddings)[0]

    #Convert to numpy arrays for manipulation
    scores = cosine_scores.cpu().numpy()
    ratings = df['normalized_rating'].values

    #Blend cosine similarity and rating
    final_scores = 0.90 * scores + 0.10 * ratings  #Adjust weighting as needed

    #Get sorted indices
    sorted_indices = final_scores.argsort()[::-1]

    #Return full sorted list (title, score) - saves other options for re-roll
    results = []
    for idx in sorted_indices:
        title = df.iloc[idx]['title']
        score = round(float(final_scores[idx]), 3)
        poster_url = df.iloc[idx].get('poster_url', None)
        results.append((title, score, poster_url))

    return results

#Function to get poster URL from TMDB
#--------------------------------------------------------
def get_movie_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
    return None

#Load semantic model
####################
model = SentenceTransformer('all-MiniLM-L6-v2')  #Small, fast, powerful

#Try loading cached data
if os.path.exists("MovieMediator/embeddings.pt") and os.path.exists("MovieMediator/preprocessed_df.pkl"):
    print("Loading cached embeddings and dataframe...")
    movie_embeddings = torch.load("MovieMediator/embeddings.pt")
    df1 = pd.read_pickle("MovieMediator/preprocessed_df.pkl")

else:
    print("First run: processing everything...")

    df1 = pd.read_csv('MovieMediator/Datasets/tmdb_5000_movies.csv')
    df2 = pd.read_csv('MovieMediator/Datasets/tmdb_5000_credits.csv')
    #Initialize the dataframe
    df1 = df1[['id', 'title', 'keywords', 'overview', 'genres', 'vote_average']]
    df1 = df1.merge(df2, left_on='title', right_on='title')
    df1 = df1.drop(columns=['movie_id'])
    df1['normalized_rating'] = df1['vote_average'] / 10  #tmdb ratings are out of 10

    #Loop converting the strings in each of the "features" columns into objects 
    #for easier manipulation
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df1[feature] = df1[feature].apply(literal_eval)

    #Define a TF-IDF Vectorizer Object. Remove english stop words - eg. 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    df1['overview'] = df1['overview'].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(df1['overview'])

    #Find the director in the 'crew' column
    df1['director'] = df1['crew'].apply(get_director)

    #Redefine features, and apply get_list to provide top 3 for each list
    features = ['cast', 'keywords', 'genres']
    for feature in features:
        df1[feature] = df1[feature].apply(get_list)

    #Apply clean_data function to your features.
    features = ['cast', 'keywords', 'director', 'genres']
    for feature in features:
        df1[feature] = df1[feature].apply(clean_data)

    df1['soup'] = df1.apply(create_soup, axis=1)

    #Clean titles just in case (safety)
    df1['title'] = df1['title'].astype(str).str.strip()

    #Preload and cache poster URLs (once)
    print("Fetching and caching poster URLs (first run only)...")
    df1['poster_url'] = df1['title'].apply(lambda title: get_movie_poster(title) or None)

    #Load semantic model and encode all movie soups
    movie_embeddings = model.encode(df1['soup'], convert_to_tensor=True)

    #Save processed data and embeddings to avoid reprocessing
    torch.save(movie_embeddings, "MovieMediator/embeddings.pt")
    df1.to_pickle("MovieMediator/preprocessed_df.pkl")

#Vectorizer for buzzword overlap fallback (not required, but can help)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_soup_matrix = vectorizer.fit_transform(df1['soup'])

df1 = df1.reset_index()

#Final Prints
#--------------------------------------------------------------------
print("Welcome to The Movie Mediator!\n")

while True:
    buzz = input("Enter descriptive words (e.g. Fantasy Fish Animation): ")
    all_results = semantic_buzzword_search(buzz, model, movie_embeddings, df1)

    page = 0
    per_page = 3

    while True:
        print("\nTop matches:")
        #returns results (0:3), (3:6), (6:9) etc
        current_results = all_results[page * per_page : (page + 1) * per_page]

        if not current_results:
            print("No more results to show.")
            break

        for title, score, poster_url in current_results:
            print(f"- {title} (score: {score})")
            if poster_url:
                print(f"Poster: {poster_url}")
            else:
                print("Poster not found.")

        command = input("\nType 'r' to reroll, 'b' to go back, 'q' to quit to main menu: ").strip().lower()
        if command == 'r':
            page += 1
        elif command == 'b':
            if page > 0:
                page -= 1
            else:
                print("You're already at the first page.")
        elif command == 'q':
            break
        else:
            print("Invalid command.")