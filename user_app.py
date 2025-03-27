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

# -------------------------------
# User Data Helpers
# -------------------------------
def clear_user_data():
    with open("./data/user_interactions.json", "w") as f:
        json.dump({"user_preferences": {}, "user_interactions": []}, f)

def save_user_data():
    with open("./data/user_interactions.json", "w") as f:
        json.dump({
            "user_preferences": st.session_state.user_preferences,
            "user_interactions": st.session_state.user_interactions
        }, f)

def load_user_data():
    try:
        with open("./data/user_interactions.json", "r") as f:
            data = json.load(f)
            return data.get("user_preferences", {}), data.get("user_interactions", [])
    except FileNotFoundError:
        return {}, []

# Clear previous session on fresh run
clear_user_data()

# -------------------------------
# Session State Init
# -------------------------------
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences, st.session_state.user_interactions = load_user_data()

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

categories = data["category"].unique()

# -------------------------------
# Preference Selection Phase
# -------------------------------
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
                        st.session_state.user_interactions.append((row["title"], "liked"))
                        save_user_data()
                        st.rerun()

                with col2:
                    if st.session_state.current_choices[like_key] == -1:
                        st.button("Dislike", key=dislike_key, disabled=True)
                    elif st.button("Dislike", key=dislike_key):
                        st.session_state.current_choices[like_key] = -1
                        st.session_state.user_interactions.append((row["title"], "disliked"))
                        save_user_data()
                        st.rerun()

        if all(value is not None for value in st.session_state.current_choices.values()):
            if st.button("Next"):
                for key, value in st.session_state.current_choices.items():
                    category = current_category
                    if value is not None:
                        st.session_state.user_preferences[category] = st.session_state.user_preferences.get(category, 0) + value

                save_user_data()
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
elif st.session_state.recommendations_ready and st.session_state.selected_broadcast is None:
    st.title("BBC Recommender System - User Interface")
    st.header("Your Recommendations")

    def load_similarity_model():
        model_path = "./models/item_similarity.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                return pickle.load(f)
        return pd.DataFrame()

    similarity_model = load_similarity_model()

    # Get user liked titles
    liked_titles = [title for title, feedback in st.session_state.user_interactions if feedback == "liked"]

    # Get similar titles based on collaborative filtering
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

    # Load trending from all synthetic users
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

    sections = {
        "For You": data.sample(10, replace=True),
        "Last Watched": data.sample(10, replace=True),
        "Might Like to Watch": data.sample(10, replace=True),
        "Trending or Popular": data[data["title"].isin(top_trending_titles)],
        "Similar Viewers Also Watched": data[data["title"].isin(similar_titles)]
    }

    for section, content in sections.items():
        st.subheader(section)
        cols = st.columns(5)

        for i, (_, row) in enumerate(content.iterrows()):
            with cols[i % 5]:
                st.image(row["image"])
                st.write(f"**{row['title']}**")
                button_key = f"view_{section}_{row['title']}"
                if st.button(f"View {row['title']}", key=button_key):
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
            if st.button(f"View {row['title']}", key=f"view_more_{row['title']}"):
                st.session_state.selected_broadcast = row.to_dict()
                st.rerun()

    if st.button("Back to Home"):
        st.session_state.selected_broadcast = None
        st.rerun()
