import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    'Movie ID': [1, 2, 3, 4, 5, 6, 7],
    'Title': ['Inception', 'The Matrix', 'The Godfather', 'The Dark Knight', 'Titanic', 'Avatar', 'The Lion King'],
    'Genres': ['Action, Sci-Fi', 'Action, Sci-Fi', 'Crime, Drama', 'Action, Crime, Drama', 'Drama, Romance', 'Action, Adventure', 'Animation, Drama']
})

genres = ['Action', 'Sci-Fi', 'Crime', 'Drama', 'Romance', 'Animation']
for genre in genres:
    movies[genre] = movies['Genres'].apply(lambda x: 1 if genre in x else 0)

user_profile = {
    'Action': 1,
    'Sci-Fi': 0,
    'Crime': 0,
    'Drama': 1,
    'Romance': 0,
    'Animation': 0
}

user_profile = pd.DataFrame([user_profile])
movie_features = movies[genres]
similarities = cosine_similarity(user_profile, movie_features)

# Show Similarity Scores
movies['Similarity'] = similarities.flatten()
top_3_movies = movies.sort_values(by='Similarity', ascending=False).head(3)

print("\nTop 3 Recommended Movies based on Content-Based Filtering:")
print(top_3_movies[['Title', 'Similarity']])

ratings_matrix = pd.DataFrame({
    'User 1': [5, 0, 0, 4, 3, 0, 0],
    'User 2': [0, 5, 0, 0, 0, 5, 0],
    'User 3': [0, 0, 5, 0, 0, 0, 5],
    'User 4': [0, 0, 0, 5, 0, 0, 0]
}, index=[1, 2, 3, 4, 5, 6, 7])

user_similarity = cosine_similarity(ratings_matrix.T)
predicted_ratings = ratings_matrix.copy()
for user in ratings_matrix.columns:
    for movie_id in ratings_matrix.index:
        if pd.isna(ratings_matrix.loc[movie_id, user]) or ratings_matrix.loc[movie_id, user] == 0:
            similar_users = user_similarity[ratings_matrix.columns.get_loc(user)]
            
            weighted_sum = 0
            similarity_sum = 0
            for other_user, similarity in zip(ratings_matrix.columns, similar_users):
                if ratings_matrix.loc[movie_id, other_user] != 0:
                    weighted_sum += similarity * ratings_matrix.loc[movie_id, other_user]
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_ratings.loc[movie_id, user] = weighted_sum / similarity_sum
            else:
                predicted_ratings.loc[movie_id, user] = 0

user_1_ratings = predicted_ratings['User 1']
unrated_movies = user_1_ratings[user_1_ratings == 0].index
top_3_recommendations_user_1 = predicted_ratings.loc[unrated_movies, 'User 1'].sort_values(ascending=False).head(3)

print("\nTop 3 Recommended Movies for User 1 based on Collaborative Filtering:")
top_3_recommendations_user_1 = top_3_recommendations_user_1.index
for movie_id in top_3_recommendations_user_1:
    print(movies[movies['Movie ID'] == movie_id]['Title'].values[0])