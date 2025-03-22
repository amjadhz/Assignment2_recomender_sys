import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

# Define file paths
USER_DATA_FILE = "./data/user_interactions.json"

# Load dataset for metadata
@st.cache_data
def load_data():
    return pd.read_csv("./data/bbc_recommender_dataset.csv")

data = load_data()

# Load user data from JSON
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {"user_preferences": {}, "user_interactions": [], "watch_counts": {}}
    
    return data.get("user_preferences", {}), data.get("user_interactions", []), data.get("watch_counts", {})

# Load user interactions
st.session_state.user_preferences, st.session_state.user_interactions, st.session_state.watch_counts = load_user_data()

# Streamlit app title
st.title("BBC Recommender System - Developer Dashboard")

# Metrics Dashboard
st.header("Metrics Dashboard - Monitoring System Performance")

if not st.session_state.user_preferences and not st.session_state.user_interactions:
    st.write("No user interaction data available yet. Please ensure users have interacted with the recommender system.")
else:
    total_likes = sum(1 for val in st.session_state.user_preferences.values() if val > 0)
    total_dislikes = sum(1 for val in st.session_state.user_preferences.values() if val < 0)
    total_watched = sum(st.session_state.watch_counts.values())

    st.write(f"### Total Likes: {total_likes}")
    st.write(f"### Total Dislikes: {total_dislikes}")
    st.write(f"### Total Watched: {total_watched}")
    
    st.write("### User Preferences by Category")
    st.dataframe(pd.DataFrame(list(st.session_state.user_preferences.items()), columns=["Category", "Preference Score"]))

    st.write("### Recent User Interactions")
    interactions_df = pd.DataFrame(st.session_state.user_interactions, columns=["Title", "Category", "Action"])
    st.dataframe(interactions_df)

# Fairness Analysis Section
st.header("Fairness Analysis")

def compute_fairness_score(watch_count):
    return 1 / (1 + watch_count)  # Less-watched content gets a boost

# Display fairness metrics
st.write("### Fairness Impact on Recommendations")

# Watch Count Distribution
st.subheader("Watch Count Distribution")
fig, ax = plt.subplots()
ax.hist(list(st.session_state.watch_counts.values()), bins=20, edgecolor="black")
ax.set_xlabel("Watch Count")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Watch Counts")
st.pyplot(fig)

# Display least-watched content
st.subheader("Least-Watched Content (Fairness Boost Candidates)")
least_watched_df = pd.DataFrame(
    [{"title": title, "watch_count": count, "fairness_score": compute_fairness_score(count)} 
     for title, count in st.session_state.watch_counts.items()]
)

if "watch_count" in least_watched_df.columns:
    least_watched_df = least_watched_df.sort_values(by="watch_count", ascending=True).head(10)

st.dataframe(least_watched_df)

# Fairness Score Distribution
st.subheader("Fairness Score Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(least_watched_df["fairness_score"], bins=10, edgecolor="black")
ax2.set_xlabel("Fairness Score")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Fairness Scores")
st.pyplot(fig2)

# Fairness Weight Adjustment
st.subheader("Adjust Fairness Weight (α)")
alpha = st.slider("Set the weight for relevance vs. fairness (0 = full fairness, 1 = full relevance)", 0.0, 1.0, 0.8, 0.05)

st.write(f"Current Fairness Weight: **{round(alpha, 2)}**")
