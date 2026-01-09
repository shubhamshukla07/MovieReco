import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    layout="wide",
)

# =========================
# LOAD PICKLE FILES
# =========================
with open("df.pkl", "rb") as f:
    df = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)

# =========================
# TITLE
# =========================
st.title("🎬 Movie Recommendation System")
st.markdown(
    "Content-based Movie Recommender using **TF-IDF + Cosine Similarity**"
)

# =========================
# INPUT
# =========================
movie_title = st.text_input(
    "Enter a movie title",
    placeholder="e.g. Toy Story, Avatar, Inception"
)

top_n = st.slider("Number of recommendations", 5, 20, 10)

# =========================
# RECOMMENDATION FUNCTION
# =========================
def get_recommendations(title, top_n=10):
    idx = indices.get(title)
    if idx is None:
        return []  # Movie not found

    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # skip the movie itself

    recommended = [(df['title'][i], score) for i, score in sim_scores]
    return recommended

# =========================
# BUTTON
# =========================
if st.button("🎯 Recommend"):
    if not movie_title.strip():
        st.warning("Please enter a movie title")
    else:
        with st.spinner("Finding similar movies..."):
            recommendations = get_recommendations(movie_title, top_n)

            if not recommendations:
                st.warning("Movie not found in database")
            else:
                st.success("Recommendations found 🎉")
                for i, (title, score) in enumerate(recommendations, start=1):
                    st.markdown(f"**{i}. {title}**  \nSimilarity score: `{score:.4f}`")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Built with ❤️ using **Streamlit + TF-IDF**")
