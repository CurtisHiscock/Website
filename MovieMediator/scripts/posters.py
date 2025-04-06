# posters.py
############

import requests

#Insert your TMDB API key here
#--------------------------------------------------------------------
TMDB_API_KEY = "73b4f8a56d4c5d4442c8b569a755c3b6"

#Function to get poster URL from TMDB
#--------------------------------------------------------------------
def get_movie_poster(title):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title
        }
        response = requests.get(url, params=params)
        data = response.json()

        #Check if results exist and grab the first poster path
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
    return None