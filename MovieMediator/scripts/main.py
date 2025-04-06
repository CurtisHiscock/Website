# main.py
#########

import torch
from sentence_transformers import SentenceTransformer
from data_loader import load_cached_data
from recommend import semantic_buzzword_search
import nltk

#Download required NLTK data (only needs to happen once)
#--------------------------------------------------------------------
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

#Load semantic model
#--------------------------------------------------------------------
print("Loading semantic model and cached data...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, powerful

#Load preprocessed data and vector embeddings
df, movie_embeddings = load_cached_data()

#Main interface
#--------------------------------------------------------------------
print("Welcome to The Movie Mediator!\n")

while True:
    buzz = input("Enter descriptive words (e.g. Fantasy Fish Animation): ")
    all_results = semantic_buzzword_search(buzz, model, movie_embeddings, df)

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