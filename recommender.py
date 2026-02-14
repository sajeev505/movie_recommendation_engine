import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np

class MovieRecommender:
    def __init__(self, movies_path='data/movies.csv', ratings_path='data/ratings.csv'):
        print("Loading data...")
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        
        # 1. Content-Based Filtering (Genres)
        # -----------------------------------
        # Simple TF-IDF on genres to find similar movies by category
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.movies['genres'] = self.movies['genres'].str.replace('|', ' ')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['genres'])
        
        # 2. Collaborative Filtering (Item-Item)
        # --------------------------------------
        # Create a User-Item Matrix
        movie_user_mat = self.ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        self.movie_to_idx = {movie: i for i, movie in enumerate(movie_user_mat.index)}
        self.idx_to_movie = {i: movie for i, movie in enumerate(movie_user_mat.index)}
        self.movie_user_sparse = csr_matrix(movie_user_mat.values)
        
        # Using NearestNeighbors for similarity
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        self.model_knn.fit(self.movie_user_sparse)
        
    def get_movie_id(self, title):
        """Helper to find movieId by title (fuzzy match)"""
        match = self.movies[self.movies['title'].str.contains(title, case=False, na=False)]
        if not match.empty:
            return match.iloc[0]['movieId']
        return None

    def get_recommendations_content(self, title, n=10):
        """Content-based recommendation using Genres"""
        # Find movie index
        try:
            target_movie = self.movies[self.movies['title'].str.contains(title, case=False)].iloc[0]
            idx = target_movie.name # The dataframe index
        except IndexError:
            return []

        # Compute cosine similarity
        cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix) 
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        return self.movies.iloc[movie_indices][['title', 'genres']].to_dict('records')

    def get_recommendations_collab(self, title, n=10):
        """Collaborative Filtering recommendation using NearestNeighbors"""
        movie_id = self.get_movie_id(title)
        if not movie_id or movie_id not in self.movie_to_idx:
            return []
            
        idx = self.movie_to_idx[movie_id]
        distances, indices = self.model_knn.kneighbors(
            self.movie_user_sparse[idx], 
            n_neighbors=n+1
        )
        
        recs = []
        for i in range(1, len(distances.flatten())):
            recommended_idx = indices.flatten()[i]
            recommended_movie_id = self.idx_to_movie[recommended_idx]
            movie_info = self.movies[self.movies['movieId'] == recommended_movie_id].iloc[0]
            recs.append({
                'title': movie_info['title'],
                'genres': movie_info['genres']
            })
            
        return recs

    def search_movies(self, query):
        results = self.movies[self.movies['title'].str.contains(query, case=False, na=False)]
        return results[['title', 'genres']].head(10).to_dict('records')
