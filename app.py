from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from recommender import MovieRecommender
import os

app = Flask(__name__)
CORS(app)

# Initialize Recommender
# Check if data exists before initializing
if os.path.exists('data/movies.csv'):
    recommender = MovieRecommender()
else:
    recommender = None
    print("WARNING: Data not found. Please run download_data.py")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    if not recommender:
        return jsonify({"error": "Server starting up or data missing"}), 503
        
    title = request.args.get('title')
    method = request.args.get('method', 'collab') # collab or content
    
    if not title:
        return jsonify({"error": "Title is required"}), 400
        
    if method == 'content':
        results = recommender.get_recommendations_content(title)
    else:
        results = recommender.get_recommendations_collab(title)
        
    if not results:
        # Fallback to content if collab fails (e.g. new movie)
        results = recommender.get_recommendations_content(title)
        
    return jsonify(results)

@app.route('/search', methods=['GET'])
def search():
    if not recommender:
        return jsonify({"error": "Data missing"}), 503
    query = request.args.get('q', '')
    results = recommender.search_movies(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
