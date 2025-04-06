from flask import Flask, render_template, send_from_directory, request, jsonify
from scripts.data_loader import load_cached_data
from scripts.recommend import semantic_buzzword_search
from scripts.embedding_model import SimpleEmbedder
import torch
import traceback
import os

# Initialize Flask app
app = Flask(__name__)

# Root route
@app.route("/")
def home():
    return render_template("index.html")

# Serve JS, CSS, and images
@app.route('/scripts/<path:path>')
def send_js(path):
    return send_from_directory('scripts', path)

@app.route('/styles/<path:path>')
def send_css(path):
    return send_from_directory('styles', path)

@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory('images', path)

# Load model and cached data on server start
try:
    print("Loading model and cached data...")
    model_path = os.path.join(os.path.dirname(__file__), "all-MiniLM-L6-v2")
    model = SimpleEmbedder(model_path)
    df, movie_embeddings = load_cached_data()
    print("Model and data loaded successfully.")
except Exception as e:
    print("Error loading model or data:", e)
    traceback.print_exc()  # This prints the full traceback to stderr.log
    model = None
    df = None
    movie_embeddings = None

# /recommend route
@app.route("/recommend", methods=["GET"])
def recommend():
    print("Request received")

    if model is None or df is None or movie_embeddings is None:
        print("Model or data not loaded properly.")
        return jsonify({"error": "Model or data failed to load."}), 500

    query = request.args.get("query", "")
    page = int(request.args.get("page", 0))
    print(f"Query: {query} | Page: {page}")

    if not query:
        print("Missing query parameter")
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        results = semantic_buzzword_search(query, model, movie_embeddings, df)
        print(f"Search returned {len(results)} results")
    except Exception as e:
        print("Search failed:", e)
        return jsonify({"error": "Search failed", "details": str(e)}), 500

    per_page = 3
    start = page * per_page
    end = start + per_page
    paged_results = results[start:end]

    return jsonify({
        "results": [
            {"title": title, "score": score, "poster": poster}
            for title, score, poster in paged_results
        ],
        "total": len(results)
    })
