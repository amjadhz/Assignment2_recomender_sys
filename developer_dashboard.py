import streamlit as st
import pandas as pd
import json

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("./data/bbc_recommender_dataset.csv")

data = load_data()

# Function to load user interactions from file
def load_user_data():
    try:
        with open("./data/user_interactions.json", "r") as f:
            data = json.load(f)
            return data.get("user_preferences", {}), data.get("user_interactions", [])
    except FileNotFoundError:
        return {}, []

# Load user data
st.session_state.user_preferences, st.session_state.user_interactions = load_user_data()

# Streamlit app title
st.title("BBC Recommender System - Developer Dashboard")

# Metrics Dashboard
st.header("Metrics Dashboard - Monitoring System Performance")

if not st.session_state.user_preferences and not st.session_state.user_interactions:
    st.write("No user interaction data available yet. Please ensure users have interacted with the recommender system.")
else:
    total_likes = sum(1 for val in st.session_state.user_preferences.values() if val > 0)
    total_dislikes = sum(1 for val in st.session_state.user_preferences.values() if val < 0)
    total_watched = sum(1 for _, action in st.session_state.user_interactions if action == "watched")
    
    st.write(f"### Total Likes: {total_likes}")
    st.write(f"### Total Dislikes: {total_dislikes}")
    st.write(f"### Total Watched: {total_watched}")
    
    st.write("### User Preferences by Category")
    st.dataframe(pd.DataFrame(list(st.session_state.user_preferences.items()), columns=["Category", "Preference Score"]))
    
    st.write("### Recent User Interactions")
    interactions_df = pd.DataFrame(st.session_state.user_interactions, columns=["Title", "Action"])
    st.dataframe(interactions_df)

st.write("### Recommender System Value Metrics")
st.write("**Transparency:** Ensuring users understand why recommendations are made.")
st.write("**Fairness:** Maintaining a balanced representation of content.")
st.write("**Diversity:** Recommending content from multiple categories.")
st.write("**Autonomy:** Allowing users to refine their preferences dynamically.")

