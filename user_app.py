import streamlit as st
import pandas as pd
import json
import os

# Define file paths
USER_DATA_FILE = "./data/user_interactions.json"

# Function to load or initialize user data (including watch counts)
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {"user_preferences": {}, "user_interactions": [], "watch_counts": {}}
    
    return data.get("user_preferences", {}), data.get("user_interactions", []), data.get("watch_counts", {})

# Function to save user interactions and watch counts
def save_user_data():
    with open(USER_DATA_FILE, "w") as f:
        json.dump({
            "user_preferences": st.session_state.user_preferences,
            "user_interactions": st.session_state.user_interactions,
            "watch_counts": st.session_state.watch_counts
        }, f)

# Initialize session states
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences, st.session_state.user_interactions, st.session_state.watch_counts = load_user_data()

if "category_index" not in st.session_state:
    st.session_state.category_index = 0

if "current_choices" not in st.session_state:
    st.session_state.current_choices = {}

if "recommendations_ready" not in st.session_state:
    st.session_state.recommendations_ready = False

if "current_items" not in st.session_state:
    st.session_state.current_items = None

if "selected_broadcast" not in st.session_state:
    st.session_state.selected_broadcast = None

# Define category similarity mapping
category_similarity = {
    "Technology": ["Science", "Innovation"],
    "Sports": ["Health", "Fitness"],
    "Politics": ["World", "Economy"],
    "Entertainment": ["Culture", "Lifestyle"],
    "Business": ["Finance", "Economy"],
    "Science": ["Technology", "Research"]
}

# Load dataset (only for metadata)
@st.cache_data
def load_data():
    return pd.read_csv("./data/bbc_recommender_dataset.csv")

data = load_data()
categories = data["category"].unique()

# Fairness Score Calculation (based on watch count stored in JSON)
def compute_fairness_score(title):
    watch_count = st.session_state.watch_counts.get(title, 0)
    return 1 / (1 + watch_count)  # Less-watched content gets a boost

# Get fairness-adjusted recommendations
def get_fair_recommendations(data):
    if "relevance_score" not in data.columns:
        data["relevance_score"] = 1  # Default relevance score if missing

    data["fairness_score"] = data["title"].apply(compute_fairness_score)
    data["adjusted_score"] = 0.8 * data["relevance_score"] + 0.2 * data["fairness_score"]
    return data.sort_values(by="adjusted_score", ascending=False)

# User Chooses Preferences
st.header("Discover BBC Content")

if not st.session_state.recommendations_ready:
    if st.session_state.category_index < len(categories):
        current_category = categories[st.session_state.category_index]
        st.subheader(f"Category: {current_category}")
        
        if st.session_state.current_items is None:
            st.session_state.current_items = data[data["category"] == current_category].sample(3, replace=True)
        
        cols = st.columns(3)
        
        for i, (_, row) in enumerate(st.session_state.current_items.iterrows()):
            with cols[i]:
                st.image(row["image"])
                st.write(f"**{row['title']}**")
                st.write(row["description"])
                
                like_key = f"like_{row['title']}"
                dislike_key = f"dislike_{row['title']}"
                
                if like_key not in st.session_state.current_choices:
                    st.session_state.current_choices[like_key] = None
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.session_state.current_choices[like_key] == 1:
                        st.button("👍", key=like_key, disabled=True, help="Liked")
                    elif st.button("👍", key=like_key):
                        st.session_state.current_choices[like_key] = 1
                        st.session_state.user_interactions.append((row["title"], "liked"))
                        save_user_data()
                        st.rerun()
                
                with col2:
                    if st.session_state.current_choices[like_key] == -1:
                        st.button("👎", key=dislike_key, disabled=True, help="Disliked")
                    elif st.button("👎", key=dislike_key):
                        st.session_state.current_choices[like_key] = -1
                        st.session_state.user_interactions.append((row["title"], "disliked"))
                        save_user_data()
                        st.rerun()
        
        if all(value is not None for value in st.session_state.current_choices.values()):
            liked_count = sum(1 for val in st.session_state.current_choices.values() if val == 1)
            disliked_count = sum(1 for val in st.session_state.current_choices.values() if val == -1)

            if st.button("Next"):
                for key, value in st.session_state.current_choices.items():
                    category = current_category
                    if value is not None:
                        st.session_state.user_preferences[category] = st.session_state.user_preferences.get(category, 0) + value
                
                save_user_data()

                # Adjust next categories based on preferences
                if liked_count == 3 and current_category in category_similarity:
                    # Insert similar categories next in queue
                    similar_categories = category_similarity[current_category]
                    categories = list(categories)
                    for cat in reversed(similar_categories):
                        if cat in categories:
                            categories.insert(st.session_state.category_index + 1, cat)
                
                elif disliked_count >= 2 and current_category in category_similarity:
                    # Remove similar categories from queue
                    categories = [cat for cat in categories if cat not in category_similarity[current_category]]
                
                # Move to the next category
                st.session_state.current_choices = {}
                st.session_state.current_items = None
                st.session_state.category_index += 1
                st.rerun()
    else:
        st.session_state.recommendations_ready = True
        st.rerun()

# Home Page after choosing preferences
elif st.session_state.selected_broadcast is None:
    st.title("BBC Recommender System - User Interface")
    st.header("Your Recommendations")

    sections = {
        "For You": get_fair_recommendations(data).head(10),
        "Trending Now": data.sample(10, replace=True),
        "Most Watched": data.sample(10, replace=True),
    }

    for section, content in sections.items():
        st.subheader(section)
        cols = st.columns(5)
        for i, (_, row) in enumerate(content.iterrows()):
            with cols[i % 5]:
                st.image(row["image"])
                st.write(f"**{row['title']}**")
                if st.button(f"View {row['title']}", key=f"view_{row['title']}"):
                    st.session_state.selected_broadcast = row.to_dict()
                    st.rerun()

# Broadcast Page
else:
    st.title(st.session_state.selected_broadcast["title"])
    st.image(st.session_state.selected_broadcast["image"])
    st.write(st.session_state.selected_broadcast["description"])

    # Like & Dislike
    like_key = f"like_{st.session_state.selected_broadcast['title']}"
    dislike_key = f"dislike_{st.session_state.selected_broadcast['title']}"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Like", key=like_key):
            st.session_state.user_interactions.append((st.session_state.selected_broadcast["title"], "liked"))
            save_user_data()
    with col2:
        if st.button("👎 Dislike", key=dislike_key):
            st.session_state.user_interactions.append((st.session_state.selected_broadcast["title"], "disliked"))
            save_user_data()

    # Increment Watch Count
    st.session_state.watch_counts[st.session_state.selected_broadcast["title"]] = \
        st.session_state.watch_counts.get(st.session_state.selected_broadcast["title"], 0) + 1
    save_user_data()

    if st.button("Back to Home"):
        st.session_state.selected_broadcast = None
        st.rerun()
        