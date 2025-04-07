# The Movie Mediator

Picking the right movie to watch can be a chore, and sometimes you spend longer looking for a movie than you do actually watching. If this sounds familiar, then **The Movie Mediator** is the tool for you. 

The Movie Mediator is a semantic movie recommendation app that suggests films based on buzzwords that **you** provide. Just tell us exactly what you're feeling and we'll recommend something great! 
Using sentence embeddings and cosine similarity, the app finds the best matches from the [TMDB_5000 Movie dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and returns them alongside poster images and confidence scores.

---

## Features
- Buzzword-based movie recommendations
- Embeds user input and movie metadata with a sentence transformer model
- Ranks results by semantic similarity + rating score
- Clean JSON output with title, score, and poster URL
- Designed for future art direction and re-rolling logic

---

## How It Works
- User inputs buzzwords like `funny zombie roadtrip`
- Model encodes input into a semantic vector
- Cosine similarity is computed against pre-embedded movie dataset
- Top results are returned as JSON via the `/recommend` API route

---

## How to Run (Locally)

### Just Clone the Repository!

You can find the repository for my website [Here](https://github.com/CurtisHiscock/Website)!
Once you've cloned that, you just need to make sure you've installed everything from requirements.txt.
<br>

```bash 
# Install dependencies 
pip install -r requirements.txt 
# Run the app 
python app.py 
```

Running app.py will give you the address to the search engine - just open that in your browser and start coming up with movie prompts!

## Extra Notes
- The button on the right is used to re-roll if you like your prompt but want some new results
- The button on the left will take you back in case you re-rolled some results you want to revisit
- The home button in the top left will take you back to [my website](https://curtishiscock.com)
