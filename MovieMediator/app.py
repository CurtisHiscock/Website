# app.py
########

from flask import Flask, request, jsonify, render_template, send_from_directory
from sentence_transformers import SentenceTransformer
from scripts.data_loader import load_cached_data
from scripts.recommend import semantic_buzzword_search
import nltk

#Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

#Initialize Flask app
app = Flask(__name__)

#Load model and data (on server start)
model = SentenceTransformer('all-MiniLM-L6-v2')
df, movie_embeddings = load_cached_data()

#Define a route for buzzword-based recommendations
#--------------------------------------------------------------------
@app.route("/recommend", methods=["GET"])
def recommend():
    query = request.args.get("query", "")
    page = int(request.args.get("page", 0))  # Default to page 0

    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    #Run the semantic search
    results = semantic_buzzword_search(query, model, movie_embeddings, df)

    #Slice results for pagination
    per_page = 3
    start = page * per_page
    end = start + per_page
    paged_results = results[start:end]

    #Format for JSON
    response = [
        {"title": title, "score": score, "poster": poster}
        for title, score, poster in paged_results
    ]

    return jsonify({
    "results": response,
    "total": len(results)
})

#Serve frontend page
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/scripts/<path:path>')
def send_js(path):
    return send_from_directory('scripts', path)

@app.route('/styles/<path:path>')
def send_css(path):
    return send_from_directory('styles', path)

@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory('images', path)

#Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)