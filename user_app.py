import streamlit as st
import pandas as pd
import json
import os

import pickle

# -------------------------------
# Load BBC Dataset
# -------------------------------
USER_DATA_FILE = "./data/user_interactions.json"

# Function to clear user interactions file
def clear_user_data():
    with open(USER_DATA_FILE, "w") as f:
        json.dump({"user_preferences": {}, "user_interactions": [], "watch_counts": {}}, f)

# ✅ Clear user data ONLY once at start of fresh session
if "initialized" not in st.session_state:
    clear_user_data()
    st.session_state.initialized = True
# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("./data/bbc_recommender_dataset_clean.csv")

data = load_data()


# -------------------------------
# User Data Helpers
# -------------------------------
def load_user_data():
    try:
        with open(USER_DATA_FILE, "r") as f:
            data = json.load(f)
            return data.get("user_preferences", {}), data.get("user_interactions", []), data.get("watch_counts", {})
    except FileNotFoundError:
        return {}, [], {}

# Function to save user interactions and watch counts
def save_user_data():
    with open(USER_DATA_FILE, "w") as f:
        json.dump({
            "user_preferences": st.session_state.user_preferences,
            "user_interactions": st.session_state.user_interactions,
            "watch_counts": st.session_state.watch_counts
        }, f)

# Why was this recommended?
def get_recommendation_reason(broadcast):
    category = broadcast.get("category", "Unknown")
    title = broadcast.get("title", "")
    liked_cats = {
        inter[1] for inter in st.session_state.user_interactions if len(inter) > 2 and inter[2] == "liked"
    }
    reason_parts = []
    if category in liked_cats:
        reason_parts.append(f"You've liked **{category}** content before.")
    if st.session_state.watch_counts.get(title, 0) == 0:
        reason_parts.append("You haven't watched this before.")
    else:
        reason_parts.append("You've shown interest in this previously.")
    return " ".join(reason_parts) or "Matches your selected preferences."

# Handle view button click - callback function
def view_broadcast(broadcast):
    st.session_state.selected_broadcast = broadcast

# Handle back button click
def go_back_home():
    st.session_state.selected_broadcast = None


# -------------------------------
# Session State Init
# -------------------------------
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

categories = data["category"].unique()

# -------------------------------
# Fairness Score Calculation
# -------------------------------
def compute_fairness_score(title):
    watch_count = st.session_state.watch_counts.get(title, 0)
    return 1 / (1 + watch_count)  # Less-watched content gets a boost

def get_fair_recommendations(data):
    if "relevance_score" not in data.columns:
        data["relevance_score"] = 1  # Default relevance score if missing

    data["fairness_score"] = data["title"].apply(compute_fairness_score)
    data["adjusted_score"] = 0.8 * data["relevance_score"] + 0.2 * data["fairness_score"]
    return data.sort_values(by="adjusted_score", ascending=False)

def get_interest_based_recommendations(data):
    # Fixed: Use the category at index 1 and check if interaction is "liked"
    liked_categories = set()
    for interaction in st.session_state.user_interactions:
        if len(interaction) > 2 and interaction[2] == "liked":  # Check for "liked" at index 2
            liked_categories.add(interaction[1])  # Category is at index 1
    
    if liked_categories:
        interest_recommendations = data[data["category"].isin(liked_categories)].sample(10, replace=True)
    else:
        interest_recommendations = data.sample(10, replace=True)
    return interest_recommendations

# Function for the refresh button that gets the next 10 recommendations
def refresh_section(section_name):
    # Initialize offset tracking in session state if not exists
    if 'recommendation_offsets' not in st.session_state:
        st.session_state.recommendation_offsets = {
            "For You": 0,
            "Interested": 0
        }
    
    current_section_data = sections[section_name].head(10)
    
    # Mark skipped recommendations in user interactions
    for i, row in current_section_data.iterrows():
        category = row['category'] if 'category' in row else 'Unknown'
        st.session_state.user_interactions.append((row['title'], category, section_name, 'skipped'))
        save_user_data()
    
    # Remove the current section's recommendations
    if section_name == "For You":
        full_recommendations = get_fair_recommendations(data)
        current_offset = st.session_state.recommendation_offsets["For You"]
        new_recommendations = full_recommendations.iloc[current_offset+10:current_offset+20]
        st.session_state.recommendation_offsets["For You"] += 10

    elif section_name == "Interested":
        full_recommendations = get_interest_based_recommendations(data)
        current_offset = st.session_state.recommendation_offsets["Interested"]
        new_recommendations = full_recommendations.iloc[current_offset+10:current_offset+20]
        st.session_state.recommendation_offsets["Interested"] += 10

    elif section_name == "Trending Now":
        new_recommendations = data.sample(10, replace=True)

    elif section_name == "Most Watched":
        new_recommendations = data.sample(10, replace=True)
    
    # Update the sections dictionary with new recommendations
    sections[section_name] = new_recommendations

# -------------------------------
# Recommendation Homepage
# -------------------------------
if not st.session_state.recommendations_ready:
    st.title("Discover BBC Content")
    
    # Initialize the genre selection state if not already done
    if "genre_selection_complete" not in st.session_state:
        st.session_state.genre_selection_complete = False
    
    if "liked_genres" not in st.session_state:
        st.session_state.liked_genres = {}
    
    # Select which categories the user likes
    if not st.session_state.genre_selection_complete:
        st.header("Select Genres You're Interested In")
        st.write("Please select at least 5 genres you like by clicking on them.")
        
        # Calculate how many genres have been selected
        liked_count = sum(1 for val in st.session_state.liked_genres.values() if val == True)
        
        # Display genres in a grid using columns
        cols_per_row = 4
        genre_rows = [categories[i:i+cols_per_row] for i in range(0, len(categories), cols_per_row)]
        
        for row in genre_rows:
            cols = st.columns(cols_per_row)
            for i, genre in enumerate(row):
                with cols[i]:
                    # Initialize genre preference if not already set
                    if genre not in st.session_state.liked_genres:
                        st.session_state.liked_genres[genre] = False
                    
                    # Store the previous state to detect changes
                    previous_state = st.session_state.liked_genres.get(genre)
                    
                    # Button and its styling
                    button_style = "primary" if st.session_state.liked_genres.get(genre) else "secondary"
                    prefix = "✓ " if st.session_state.liked_genres.get(genre) else ""
                    if st.button(f"{prefix}{genre}", key=f"genre_btn_{genre}", type=button_style):
                        # Toggle the selection state
                        st.session_state.liked_genres[genre] = not previous_state
                        st.rerun()
        
        # Continue button - only enabled if at least 5 genres are selected
        if liked_count >= 5:
            if st.button("Continue to Content Selection"):

                # Record all genre selections before continuing
                for genre in categories:
                    is_selected = st.session_state.liked_genres.get(genre, True)
                    preference_value = 1 if is_selected else -1
                    st.session_state.user_preferences[genre] = st.session_state.user_preferences.get(genre, 0) + preference_value
                
                # Save all the data at once
                save_user_data()
                
                # Filter categories to only include liked genres
                st.session_state.filtered_categories = [
                    genre for genre in categories if st.session_state.liked_genres.get(genre) == True
                ]
                st.session_state.category_index = 0
                st.session_state.genre_selection_complete = True
                st.rerun()
        else:
            st.button("Continue to Content Selection", disabled=True, help="Please select at least 5 genres you like")
    
    # Rating content for liked categories
    else:
        if st.session_state.category_index < len(st.session_state.filtered_categories):
            current_category = st.session_state.filtered_categories[st.session_state.category_index]
            st.header("What type of content do you like?")
            st.write("Before we show you your recommendations, please tell us more about what you like to watch:")
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
                        if st.session_state.current_choices.get(like_key) == 1:
                            st.button("👍", key=like_key, disabled=True, help="Liked")
                        elif st.button("👍", key=like_key):
                            st.session_state.current_choices[like_key] = 1
                            st.session_state.user_interactions.append((row["title"], row["category"], "liked"))
                            save_user_data()
                            st.rerun()
                    
                    with col2:
                        if st.session_state.current_choices.get(like_key) == -1:
                            st.button("👎", key=dislike_key, disabled=True, help="Disliked")
                        elif st.button("👎", key=dislike_key):
                            st.session_state.current_choices[like_key] = -1
                            st.session_state.user_interactions.append((row["title"], row["category"], "disliked"))
                            save_user_data()
                            st.rerun()
            
            if all(value is not None for value in st.session_state.current_choices.values()):
                liked_count = sum(1 for val in st.session_state.current_choices.values() if val == 1)
                disliked_count = sum(1 for val in st.session_state.current_choices.values() if val == -1)
                
                if st.button("Next"):                   
                    save_user_data()
                    
                    # Adjust next categories based on preferences
                    if liked_count == 3 and current_category in category_similarity:
                        # Insert similar categories next in queue if they are in liked genres
                        similar_categories = category_similarity[current_category]
                        filtered_categories_list = list(st.session_state.filtered_categories)
                        for cat in reversed(similar_categories):
                            if cat in filtered_categories_list:
                                filtered_categories_list.insert(st.session_state.category_index + 1, cat)
                        st.session_state.filtered_categories = filtered_categories_list
                    
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
    liked_titles = [
    interaction[0]  # title
    for interaction in st.session_state.user_interactions
    if len(interaction) > 2 and interaction[2] == "liked"
]

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
        "For You": get_fair_recommendations(data).head(10),
        "Interested": get_interest_based_recommendations(data),
        "Trending Now": data.sample(10, replace=True),
        "Most Watched": data.sample(10, replace=True)
    }
    
    for section_name, content in sections.items():

        # Create a row with section title and refresh button
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.subheader(section_name)
        with col2:
            refresh_key = f"refresh_{section_name}"
            st.button(
                "🔄 Refresh", 
                key=refresh_key,
                on_click=refresh_section,
                args=(section_name,)
            )
        
        cols = st.columns(5)

        for i, (_, row) in enumerate(content.iterrows()):
            with cols[i % 5]:
                st.image(row["image"])
                st.write(f"**{row['title']}**")
                
                # Display like/dislike status if available
                for interaction in st.session_state.user_interactions:
                    if len(interaction) > 2 and interaction[0] == row["title"]:
                        if interaction[2] == "liked":
                            st.write("👍 Liked")
                        elif interaction[2] == "disliked":
                            st.write("👎 Disliked")

                        break
                
                button_key = f"view_{section_name}_{i}"
                st.button(
                    "View", 
                    key=button_key,
                    on_click=view_broadcast,
                    args=(row.to_dict(),)
                )

        st.markdown("<hr style='margin-top:2rem; margin-bottom:2rem; border:1px solid #ddd;'/>", unsafe_allow_html=True)

# -------------------------------
# Broadcast Page
# -------------------------------
else:
    try:
        st.title(st.session_state.selected_broadcast["title"])
        st.image(st.session_state.selected_broadcast["image"])
        st.write(st.session_state.selected_broadcast["description"])

        # ✅ SHOWING WHY IT WAS RECOMMENDED
        st.write(f"🔍 **Why recommended:** {get_recommendation_reason(st.session_state.selected_broadcast)}")

        # Check if user has already liked or disliked this content
        user_action = None
        for interaction in st.session_state.user_interactions:
            if len(interaction) > 2 and interaction[0] == st.session_state.selected_broadcast["title"]:
                user_action = interaction[2]  # Action is at index 2
                break
        
        like_key = f"like_broadcast_{st.session_state.selected_broadcast['title']}"
        dislike_key = f"dislike_broadcast_{st.session_state.selected_broadcast['title']}"

        col1, col2 = st.columns(2)
        with col1:
            like_disabled = user_action == "liked"
            if st.button("👍 Like", key=like_key, disabled=like_disabled):

                # Remove any existing interactions for this content
                st.session_state.user_interactions = [
                    interaction for interaction in st.session_state.user_interactions 
                    if len(interaction) <= 2 or interaction[0] != st.session_state.selected_broadcast["title"]
                ]
                st.session_state.user_interactions.append((
                    st.session_state.selected_broadcast["title"], 
                    st.session_state.selected_broadcast["category"], 
                    "liked"
                ))
                save_user_data()
                st.rerun()
            if like_disabled:
                st.write("You liked this content")
        
        with col2:
            dislike_disabled = user_action == "disliked"
            if st.button("👎 Dislike", key=dislike_key, disabled=dislike_disabled):
                # Remove any existing interactions for this content
                st.session_state.user_interactions = [
                    interaction for interaction in st.session_state.user_interactions 
                    if len(interaction) <= 2 or interaction[0] != st.session_state.selected_broadcast["title"]
                ]
                st.session_state.user_interactions.append((
                    st.session_state.selected_broadcast["title"], 
                    st.session_state.selected_broadcast["category"], 
                    "disliked"
                ))
                save_user_data()
                st.rerun()
            if dislike_disabled:
                st.write("You disliked this content")

                # Increment Watch Count
        st.session_state.watch_counts[st.session_state.selected_broadcast["title"]] = \
            st.session_state.watch_counts.get(st.session_state.selected_broadcast["title"], 0) + 1
        save_user_data()
        
        st.markdown("<hr style='margin-top:3rem; margin-bottom:2rem; border:2px solid #ccc;'/>", unsafe_allow_html=True)

        st.subheader("More Recommendations")
        more_recommendations = data.sample(5, replace=True)  

        cols = st.columns(5)
        for i, (_, row) in enumerate(more_recommendations.iterrows()):
            with cols[i % 5]:
                st.image(row["image"])
                st.write(f"**{row['title']}**")

                
                # Display like/dislike status if available
                for interaction in st.session_state.user_interactions:
                    if len(interaction) > 2 and interaction[0] == row["title"]:
                        if interaction[2] == "liked":
                            st.write("👍 Liked")
                        elif interaction[2] == "disliked":
                            st.write("👎 Disliked")
                        break

                if st.button("View", key=f"view_more_reco_{i}"):
                    st.session_state.selected_broadcast = row.to_dict()
                    st.rerun()

        if st.button("🏠 Back to Home"):
            st.session_state.selected_broadcast = None
            st.rerun()

    except Exception as e:
        st.error(f"Something went wrong: {e}")
