import streamlit as st
import requests

# =========================
# CONFIG
# =========================
FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    layout="wide",
)

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
# BUTTON
# =========================
if st.button("🎯 Recommend"):
    if not movie_title.strip():
        st.warning("Please enter a movie title")
    else:
        with st.spinner("Finding similar movies..."):
            try:
                response = requests.get(
                    f"{FASTAPI_URL}/recommend/tfidf",
                    params={
                        "title": movie_title,
                        "top_n": top_n
                    },
                    timeout=20
                )

                if response.status_code != 200:
                    st.error("Movie not found or API error")
                else:
                    recommendations = response.json()

                    if not recommendations:
                        st.warning("No recommendations found")
                    else:
                        st.success("Recommendations found 🎉")

                        for i, rec in enumerate(recommendations, start=1):
                            st.markdown(
                                f"**{i}. {rec['title']}**  \n"
                                f"Similarity score: `{rec['score']:.4f}`"
                            )

            except Exception as e:
                st.error(f"Error connecting to API: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "Built with ❤️ using **FastAPI + Streamlit + TF-IDF**"
)
