import streamlit as st
import pandas as pd
import json
import os
import pickle

# -------------------------------
# Load BBC Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("./data/bbc_recommender_dataset_clean.csv")

data = load_data()

USER_DATA_FILE = "./data/user_interactions.json"

# -------------------------------
# User Data Helpers
# -------------------------------
def clear_user_data():
    with open(USER_DATA_FILE, "w") as f:
        json.dump({"user_preferences": {}, "user_interactions": [], "watch_counts": {}}, f)

def save_user_data():
    with open(USER_DATA_FILE, "w") as f:
        json.dump({
            "user_preferences": st.session_state.user_preferences,
            "user_interactions": st.session_state.user_interactions,
            "watch_counts": st.session_state.watch_counts
        }, f)

def load_user_data():
    try:
        with open(USER_DATA_FILE, "r") as f:
            data = json.load(f)
            return data.get("user_preferences", {}), data.get("user_interactions", []), data.get("watch_counts", {})
    except FileNotFoundError:
        return {}, [], {}

# -------------------------------
# Session State Init
# -------------------------------
clear_user_data()

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

if "genre_selection_complete" not in st.session_state:
    st.session_state.genre_selection_complete = False

if "liked_genres" not in st.session_state:
    st.session_state.liked_genres = {}

categories = data["category"].unique()

# -------------------------------
# Genre Selection
# -------------------------------
st.header("Discover BBC Content")

if not st.session_state.recommendations_ready:
    if not st.session_state.genre_selection_complete:
        st.subheader("Select Genres You're Interested In")
        st.write("Please select at least 5 genres you like by clicking on them.")

        liked_count = sum(1 for val in st.session_state.liked_genres.values() if val)
        cols_per_row = 4
        genre_rows = [categories[i:i+cols_per_row] for i in range(0, len(categories), cols_per_row)]

        for row in genre_rows:
            cols = st.columns(cols_per_row)
            for i, genre in enumerate(row):
                with cols[i]:
                    if genre not in st.session_state.liked_genres:
                        st.session_state.liked_genres[genre] = False

                    previous_state = st.session_state.liked_genres.get(genre)
                    button_style = "primary" if previous_state else "secondary"
                    prefix = "✓ " if previous_state else ""
                    if st.button(f"{prefix}{genre}", key=f"genre_btn_{genre}", type=button_style):
                        st.session_state.liked_genres[genre] = not previous_state
                        st.rerun()

        if liked_count >= 5:
            if st.button("Continue to Content Selection"):
                for genre in categories:
                    is_selected = st.session_state.liked_genres.get(genre, True)
                    preference_value = 1 if is_selected else -1
                    st.session_state.user_preferences[genre] = st.session_state.user_preferences.get(genre, 0) + preference_value
                save_user_data()
                st.session_state.filtered_categories = [genre for genre in categories if st.session_state.liked_genres.get(genre)]
                st.session_state.category_index = 0
                st.session_state.genre_selection_complete = True
                st.rerun()
        else:
            st.button("Continue to Content Selection", disabled=True)

    elif st.session_state.category_index < len(st.session_state.filtered_categories):
        current_category = st.session_state.filtered_categories[st.session_state.category_index]
        st.subheader(f"Category: {current_category}")

        if st.session_state.current_items is None:
            st.session_state.current_items = data[data["category"] == current_category].sample(3, replace=True)

        cols = st.columns(3)
        for i, (_, row) in enumerate(st.session_state.current_items.iterrows()):
            with cols[i]:
                st.image(row["image"])
                st.write(f"**{row['title']}**")
                st.write(row["description"])

                unique_id = f"{row['title']}_{i}_{st.session_state.category_index}"
                like_key = f"like_{unique_id}"
                dislike_key = f"dislike_{unique_id}"

                if like_key not in st.session_state.current_choices:
                    st.session_state.current_choices[like_key] = None

                col1, col2 = st.columns(2)
                with col1:
                    if st.session_state.current_choices[like_key] == 1:
                        st.button("Like", key=like_key, disabled=True)
                    elif st.button("Like", key=like_key):
                        st.session_state.current_choices[like_key] = 1
                        st.session_state.user_interactions.append((row["title"], row["category"], "liked"))
                        save_user_data()
                        st.rerun()

                with col2:
                    if st.session_state.current_choices[like_key] == -1:
                        st.button("Dislike", key=dislike_key, disabled=True)
                    elif st.button("Dislike", key=dislike_key):
                        st.session_state.current_choices[like_key] = -1
                        st.session_state.user_interactions.append((row["title"], row["category"], "disliked"))
                        save_user_data()
                        st.rerun()

        if all(v is not None for v in st.session_state.current_choices.values()):
            if st.button("Next"):
                st.session_state.current_choices = {}
                st.session_state.current_items = None
                st.session_state.category_index += 1
                st.rerun()

    else:
        st.session_state.recommendations_ready = True
        st.rerun()

# -------------------------------
# Recommendation Homepage
# -------------------------------
elif st.session_state.selected_broadcast is None:
    st.title("BBC Recommender System - User Interface")
    st.header("Your Recommendations")

    def load_similarity_model():
        model_path = "./models/item_similarity.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                return pickle.load(f)
        return pd.DataFrame()

    similarity_model = load_similarity_model()

    liked_titles = [interaction[0] for interaction in st.session_state.user_interactions if len(interaction) > 2 and interaction[2] == "liked"]

    def get_similar_items(user_likes, model, n=10):
        if model.empty or not user_likes:
            return []
        similarity_scores = pd.Series(dtype=float)
        for title in user_likes:
            if title in model.index:
                similarity_scores = similarity_scores.add(model[title], fill_value=0)
        for title in user_likes:
            if title in similarity_scores:
                similarity_scores.pop(title)
        return similarity_scores.sort_values(ascending=False).head(n).index.tolist()

    similar_titles = get_similar_items(liked_titles, similarity_model, n=10)

    def load_all_user_jsons(path="data/user_json_profiles"):
        interaction_counts = {}
        try:
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    with open(os.path.join(path, filename), "r") as f:
                        user_data = json.load(f)
                        for title, _, feedback in user_data.get("user_interactions", []):
                            if feedback == "liked":
                                interaction_counts[title] = interaction_counts.get(title, 0) + 1
                        for title, count in user_data.get("watch_counts", {}).items():
                            interaction_counts[title] = interaction_counts.get(title, 0) + count
        except Exception as e:
            st.warning(f"Could not load user JSONs: {e}")
        return interaction_counts

    trending_scores = load_all_user_jsons()
    top_trending_titles = sorted(trending_scores.items(), key=lambda x: x[1], reverse=True)
    top_trending_titles = [title for title, _ in top_trending_titles[:10]]

    # Load diversity data (Black British & British Asian content)
    diverse_data = pd.read_csv("data/diverse_data.csv")

    sections = {
        "For You": data.sample(10, replace=True),
        "Last Watched": data.sample(10, replace=True),
        "Might Like to Watch": data.sample(10, replace=True),
        "Trending or Popular": data[data["title"].isin(top_trending_titles)],
        "Similar Viewers Also Watched": data[data["title"].isin(similar_titles)],
        "Diversity Spotlight": diverse_data.sample(5, replace=True)
    }

    for section, content in sections.items():
        st.subheader(section)
        cols = st.columns(5)
        for i, (_, row) in enumerate(content.iterrows()):
            with cols[i % 5]:
                st.image(row["image"])
                st.write(f"**{row['title']}**")
                button_key = f"view_{section}_{i}"
                if st.button("View", key=button_key):
                    st.session_state.selected_broadcast = row.to_dict()
                    st.rerun()

# -------------------------------
# Broadcast Page
# -------------------------------
else:
    st.title(st.session_state.selected_broadcast["title"])
    st.image(st.session_state.selected_broadcast["image"])
    st.write(st.session_state.selected_broadcast["description"])

    like_key = f"like_{st.session_state.selected_broadcast['title']}"
    dislike_key = f"dislike_{st.session_state.selected_broadcast['title']}"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Like", key=like_key):
            st.session_state.user_interactions.append((st.session_state.selected_broadcast["title"], "liked"))
            save_user_data()
    with col2:
        if st.button("Dislike", key=dislike_key):
            st.session_state.user_interactions.append((st.session_state.selected_broadcast["title"], "disliked"))
            save_user_data()

    st.subheader("More Recommendations")
    more_recommendations = data.sample(10, replace=True)
    cols = st.columns(5)
    for i, (_, row) in enumerate(more_recommendations.iterrows()):
        with cols[i % 5]:
            st.image(row["image"])
            st.write(f"**{row['title']}**")
            if st.button("View", key=f"view_more_{i}"):
                st.session_state.selected_broadcast = row.to_dict()
                st.rerun()

    if st.button("Back to Home"):
        st.session_state.selected_broadcast = None
        st.rerun()

