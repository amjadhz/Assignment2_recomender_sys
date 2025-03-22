import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# Create tabs for different sections
tab1, tab2 = st.tabs(["Overall", "Fairness Analysis"])

with tab1:
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

        # Pie Chart - Categories Liked
        st.subheader("Categories Liked")
        liked_categories = {k: v for k, v in st.session_state.user_preferences.items() if v > 0}
        if liked_categories:
            fig_like, ax_like = plt.subplots()
            ax_like.pie(liked_categories.values(), labels=liked_categories.keys(), autopct="%.1f%%", colors=sns.color_palette("pastel"))
            ax_like.set_title("Liked Categories Distribution")
            st.pyplot(fig_like)
        else:
            st.write("No liked categories available.")

        # Pie Chart - Categories Disliked
        st.subheader("Categories Disliked")
        disliked_categories = {k: abs(v) for k, v in st.session_state.user_preferences.items() if v < 0}
        if disliked_categories:
            fig_dislike, ax_dislike = plt.subplots()
            ax_dislike.pie(disliked_categories.values(), labels=disliked_categories.keys(), autopct="%.1f%%", colors=sns.color_palette("pastel"))
            ax_dislike.set_title("Disliked Categories Distribution")
            st.pyplot(fig_dislike)
        else:
            st.write("No disliked categories available.")

with tab2:
    # Fairness Analysis Section
    st.header("Fairness Analysis")

    # 1. Bar Chart - User Preferences by Category
    st.subheader("User Preferences by Category")
    st.write("This visualization shows how the user rates different content categories. Higher values indicate a stronger preference, while lower values suggest dislike.")
    fig1, ax1 = plt.subplots()
    sns.barplot(x=list(st.session_state.user_preferences.keys()), y=list(st.session_state.user_preferences.values()), ax=ax1, palette="coolwarm")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Preference Score")
    ax1.set_title("User Preferences Distribution")
    st.pyplot(fig1)

    # 2. Stacked Bar Chart - Interaction Breakdown by Category
    st.subheader("User Interactions by Category")
    st.write("This chart illustrates how users interact with content across different categories. Green bars represent liked content, while red bars indicate disliked content.")
    interactions_df = pd.DataFrame(st.session_state.user_interactions, columns=["Title", "Category", "Action"])
    interaction_counts = interactions_df.groupby(["Category", "Action"]).size().unstack(fill_value=0)
    fig2, ax2 = plt.subplots()
    interaction_counts.plot(kind="bar", stacked=True, figsize=(10, 5), color=["red", "green"], ax=ax2)
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Count")
    ax2.set_title("User Interactions Distribution")
    st.pyplot(fig2)

    # 3. Histogram - Diversity in Recommendations
    st.subheader("Distribution of Categories in User Interactions")
    st.write("This histogram helps assess the diversity of content recommendations. A well-balanced distribution indicates fairness in recommendations.")
    fig3, ax3 = plt.subplots()
    sns.histplot(interactions_df["Category"], bins=len(interactions_df["Category"].unique()), kde=False, color="steelblue", ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.set_xlabel("Category")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Diversity of Recommended Categories")
    st.pyplot(fig3)

    # 4. Scatter Plot - Popularity Bias in Recommendations
    st.subheader("Watch Count Distribution of Recommendations")
    st.write("This scatter plot helps identify whether the recommender system favors popular content. A fair system should balance popular and less-known content recommendations.")
    watch_counts_df = pd.DataFrame(st.session_state.watch_counts.items(), columns=["Title", "Watch Count"])
    watch_counts_df = watch_counts_df.sort_values(by="Watch Count", ascending=False)
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=range(len(watch_counts_df)), y="Watch Count", data=watch_counts_df, color="red", alpha=0.7, ax=ax4)
    ax4.set_xlabel("Title Index (Sorted by Popularity)")
    ax4.set_ylabel("Watch Count")
    ax4.set_title("Popularity Bias in Recommendations")
    st.pyplot(fig4)

    # Fairness Weight Adjustment
    st.subheader("Adjust Fairness Weight (α)")
    alpha = st.slider("Set the weight for relevance vs. fairness (0 = full fairness, 1 = full relevance)", 0.0, 1.0, 0.8, 0.05)
    st.write(f"Current Fairness Weight: **{round(alpha, 2)}**")
