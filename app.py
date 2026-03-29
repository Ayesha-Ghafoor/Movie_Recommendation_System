import pickle
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load data
# ---------------------------
movies = pickle.load(open('model/movie_list.pkl', 'rb'))

# ---------------------------
# Vectorization
# ---------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# ---------------------------
# Similarity
# ---------------------------
similarity = cosine_similarity(vectors)

movie_list = movies['title'].values

# ---------------------------
# Fetch Poster Function
# ---------------------------
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()

    poster_path = data.get('poster_path', None)
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        return ""

# ---------------------------
# Recommendation Function
# ---------------------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movie_names, recommended_movie_posters

# ---------------------------
# UI
# ---------------------------
st.header('🎬 Movie Recommender System')

selected_movie = st.selectbox(
    "Type or select a movie from dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    cols = [col1, col2, col3, col4, col5]

    for i in range(5):
        with cols[i]:
            st.text(recommended_movie_names[i])
            st.image(recommended_movie_posters[i])




