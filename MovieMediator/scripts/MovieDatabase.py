import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

#print(os.getcwd()) #Where the code is running from

df1=pd.read_csv('MovieMediator/Datasets/tmdb_5000_movies.csv')
df2=pd.read_csv('MovieMediator/Datasets/tmdb_5000_credits.csv')
#Initialize the dataframe
df1 = df1[['id', 'title', 'keywords', 'overview', 'genres']]
df1 = df1.merge(df2, left_on='title', right_on='title')
df1 = df1.drop(columns=['movie_id'])

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

#Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#Construct a reverse map of indices and movie titles
indices = pd.Series(df1.index, index=df1['title']).drop_duplicates()
#Create a case-insensitive version of indices
#indices_lower = {title.lower(): index for title, index in indices.items()}
#print(indices_lower)
#print(indices)



#Function that takes in movie title as input and outputs most similar movies
#---------------------------------------------------------------------------
def get_recommendations(title, cosine_sim=cosine_sim):
    #Get the index of the movie matching the title
    idx = indices[title]

    #Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    #Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    #Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    #Return the top 10 most similar movies
    return df1['title'].iloc[movie_indices]

#Function that gets the directors name, NaN if not listed
#--------------------------------------------------------
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

#Find the director in the 'crew' column
df1['director'] = df1['crew'].apply(get_director)

#Function that returns top 3 elements or full list depending on which is larger
#----------------------------------------------------------------
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. 
        #If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

#Redefine features, and apply get_list to provide top 3 for each list
features = ['cast', 'keywords', 'genres']
for feature in features:
    df1[feature] = df1[feature].apply(get_list)

#Function to convert all strings to lower case and strip names of spaces
#------------------------------------------------------------------------
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
#Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    df1[feature] = df1[feature].apply(clean_data)

#Metadata soup
#-------------
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df1['soup'] = df1.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df1['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df1 = df1.reset_index()
indices = pd.Series(df1.index, index=df1['title'])

#Final Prints
#------------
print("Please enter the name of a movie to give us a "
      "base for your recommendations:")
#loops the process (just for testing right now)
while True:
    movietitle = input()
    #movietitle = movietitle.lower()
    
    if movietitle in indices:
        print(get_recommendations(movietitle), cosine_sim2)
        print("\n")
        print("Please enter another movie:")
    else:
        print ("Movie not found. Please try again:")


#Print functions
#---------------

#print(df1[['title', 'cast', 'director', 'keywords', 'genres']].head())
#print(df1['title'].head(3))
#print(df1['overview'].head(3))
