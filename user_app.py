import streamlit as st
import pandas as pd
import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#LAYOUT
st.set_page_config(layout="wide", page_title="BBC Recommender System")

st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 1800px;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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

# Load dataset
@st.cache_data
def load_diverse_data():
    return pd.read_csv("./data/diverse_data.csv")

data = load_data()
diverse_data = load_diverse_data()


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
    # Create a copy to avoid modifying the original
    recommendations = data.copy()
    
    if "relevance_score" not in recommendations.columns:
        recommendations["relevance_score"] = 1.0  
    
    recommendations["fairness_score"] = recommendations["title"].apply(compute_fairness_score)
    recommendations["adjusted_score"] = 0.8 * recommendations["relevance_score"] + 0.2 * recommendations["fairness_score"]
    return recommendations.sort_values(by="adjusted_score", ascending=False)

# -------------------------------
# Get Recommendations based on interested categories
# -------------------------------

def get_interest_based_recommendations(data):
    liked_categories = set()
    for interaction in st.session_state.user_interactions:
        if len(interaction) > 2 and interaction[2] == "liked":  
            liked_categories.add(interaction[1]) 
    
    if liked_categories:
        interest_recommendations = data[data["category"].isin(liked_categories)]
    else:
        interest_recommendations = data.sample(10, replace=True)
    return interest_recommendations

# -------------------------------
# Collaborative Filtering Functions
# -------------------------------

@st.cache_resource
def load_similarity_model():
    model_path = "./models/item_similarity.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return pd.DataFrame()

similarity_model = load_similarity_model()

def get_similar_items(user_likes, user_watches, model, n=10):
    if model.empty or (not user_likes and not user_watches):
        return []  # Return empty list if no data

    similarity_scores = pd.Series(0.0, index=model.columns)

    # Process liked titles (weight = 1.0)
    for title in user_likes:
        if title in model.index:
            similarity_scores = similarity_scores.add(model[title] * 1.0, fill_value=0)

    # Process watched titles (weight = 0.5 per watch count)
    for title, count in user_watches.items():
        if title in model.index:
            similarity_scores = similarity_scores.add(model[title] * (0.5 * count), fill_value=0)

    # Remove already liked/watched titles
    for title in user_likes + list(user_watches.keys()):
        if title in similarity_scores:
            similarity_scores = similarity_scores.drop(title)

    # If we have no scores left, return empty list
    if len(similarity_scores) == 0:
        return []
        
    return similarity_scores.sort_values(ascending=False).head(n).index.tolist()

# -------------------------------
# Trending Items Functions
# -------------------------------

# Load trending from all synthetic users
def load_all_user_jsons(path="data/user_json_profiles"):
    interaction_counts = {}
    try:
        for filename in os.listdir(path):
            if filename.endswith(".json"):
                with open(os.path.join(path, filename), "r") as f:
                    user_data = json.load(f)

                    # Loop through interactions
                    for i in user_data.get("user_interactions", []):
                        title = i[0]
                        feedback = i[2]
                        if feedback == "liked":
                            interaction_counts[title] = interaction_counts.get(title, 0) + 1

                    # Loop through watch counts
                    for title, count in user_data.get("watch_counts", {}).items():
                        interaction_counts[title] = interaction_counts.get(title, 0) + count
    except Exception as e:
        st.warning(f"Could not load user JSONs: {e}")
    return interaction_counts


@st.cache_data
def load_all_user_jsons_cached(path="data/user_json_profiles"):
    interaction_counts = {}
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                user_data = json.load(f)
                for i in user_data.get("user_interactions", []):
                    title = i[0]
                    feedback = i[2]
                    if feedback == "liked":
                        interaction_counts[title] = interaction_counts.get(title, 0) + 1
                for title, count in user_data.get("watch_counts", {}).items():
                    interaction_counts[title] = interaction_counts.get(title, 0) + count
    return interaction_counts


@st.cache_data
def load_all_user_watch_counts_cached(path="data/user_json_profiles"):
    watch_counts = {}
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r") as f:
                user_data = json.load(f)
                for title, count in user_data.get("watch_counts", {}).items():
                    watch_counts[title] = watch_counts.get(title, 0) + count
    return watch_counts

trending_scores = load_all_user_jsons_cached()
watch_counts_data = load_all_user_watch_counts_cached()


top_trending_titles = sorted(trending_scores.items(), key=lambda x: x[1], reverse=True)
top_trending_titles = [title for title, _ in top_trending_titles]

# -------------------------------
# Most Watched Items Function
# -------------------------------

def load_all_user_watch_counts(path="data/user_json_profiles"):
    watch_counts = {}
    try:
        for filename in os.listdir(path):
            if filename.endswith(".json"):
                with open(os.path.join(path, filename), "r") as f:
                    user_data = json.load(f)
                    for title, count in user_data.get("watch_counts", {}).items():
                        watch_counts[title] = watch_counts.get(title, 0) + count
    except Exception as e:
        st.warning(f"Could not load user JSONs: {e}")
    return watch_counts

# Get the top watched titles
watch_counts_data = load_all_user_watch_counts()
top_watched_titles = sorted(watch_counts_data.items(), key=lambda x: x[1], reverse=True)
top_watched_titles = [title for title, _ in top_watched_titles]

# -------------------------------
# Content Based Filtering Functions
# -------------------------------

# Function to build similarity matrix of items using TF-IDF for 'description' column
def build_similarity_matrix(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['description'])  
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

# Function to get content based recommendations using matrix from function above - based on 'description' column
def get_content_based_recommendations(data, user_interactions):
    # Ensure user_interactions is properly formatted
    if isinstance(user_interactions, dict) and "user_interactions" in user_interactions:
        # Convert user_interactions list to DataFrame
        user_interactions_df = pd.DataFrame(
            user_interactions["user_interactions"], 
            columns=["title", "category", "interaction"] 
            if len(user_interactions["user_interactions"]) > 0 and len(user_interactions["user_interactions"][0]) == 3
            else ["title", "category", "section", "interaction"]
        )
    else:
        user_interactions_df = pd.DataFrame(user_interactions, columns=["title", "category", "interaction"])
    
    # Ensure the interaction column exists
    if "interaction" not in user_interactions_df.columns:
        user_interactions_df["interaction"] = "viewed"
        
    # Get liked, disliked, and skipped items
    liked_items = user_interactions_df[user_interactions_df["interaction"] == "liked"]["title"].tolist()
    disliked_items = user_interactions_df[user_interactions_df["interaction"] == "disliked"]["title"].tolist()
    skipped_items = user_interactions_df[user_interactions_df["interaction"] == "skipped"]["title"].tolist()
    
    # Get watched items and their watch counts
    watch_counts = user_interactions.get("watch_counts", {}) if isinstance(user_interactions, dict) else {}
    watched_items = list(watch_counts.keys())

    # Build similarity matrix using the description column
    similarity_matrix = build_similarity_matrix(data)

    # Items to recommend
    recommended_items = set()
    # Items to exclude from recommendations - disliked + skipped
    excluded_items = set(disliked_items + skipped_items)

    # Find items similar to disliked items and exclude them
    for title in disliked_items:
        if title in data['title'].values:
            item_index = data[data['title'] == title].index[0]
            similar_indices = similarity_matrix[item_index].argsort()[-11:-1]  # Top 10 similar items
            excluded_items.update(data.iloc[similar_indices]['title'].values)

    # Find similar items to liked movies (higher weight)
    for title in liked_items:
        if title in data['title'].values:
            item_index = data[data['title'] == title].index[0]
            similar_indices = similarity_matrix[item_index].argsort()[-11:-1]  # Top 10 similar items
            recommended_items.update(data.iloc[similar_indices]['title'].values)

    # Find similar items to watched movies (lower weight)
    for title in watched_items:
        if title in data['title'].values:
            item_index = data[data['title'] == title].index[0]
            similar_indices = similarity_matrix[item_index].argsort()[-6:-1]  # Top 5 similar items (lower weight)
            recommended_items.update(data.iloc[similar_indices]['title'].values)

    # Remove any items that the user disliked, skipped, or are similar to disliked items
    recommended_items -= excluded_items
    
    # If no recommendations, return empty DataFrame with correct structure
    if not recommended_items:
        return data.head(0)

    return data[data['title'].isin(recommended_items)]

# -------------------------------
# Refresh Button Function
# -------------------------------

def refresh_section(section_name):
    if 'current_rec_type' not in st.session_state:
        st.session_state.current_rec_type = "Recommendations based on what people like me watch"
    
    current_rec_type = st.session_state.current_rec_type
    
    if 'sections' not in st.session_state:
        st.session_state.sections = {
            "For You": filtered_recommendations.head(10),
            "Interested": filtered_recommendations.head(10),
            "Trending Now": data[data["title"].isin(top_trending_titles)].head(10),
            "Most Watched": data[data["title"].isin(top_watched_titles)].head(10),
            "Random Selection": data.sample(10, replace=True),
            "Diversity Spotlight": diverse_data.sample(10, replace=False)
        }

    if 'recommendation_offsets' not in st.session_state:
        st.session_state.recommendation_offsets = {"For You": 0, "Interested": 0}

    # Skip refreshing static sections
    if section_name in ["Trending Now", "Most Watched", "Diversity Spotlight"]:
        return

    try:
        for _, row in st.session_state.sections[section_name].iterrows():
            st.session_state.user_interactions.append((row['interaction'], "skipped"))
    except:
        pass

    save_user_data()

    if section_name in ["For You", "Interested"]:
        full_recommendations = generate_recommendations(current_rec_type, data, st.session_state.user_interactions).drop_duplicates()

        if section_name == "Interested":
            full_recommendations = get_interest_based_recommendations(full_recommendations)

        offset = st.session_state.recommendation_offsets.get(section_name, 0)
        if offset + 10 >= len(full_recommendations):
            offset = 0

        end_offset = min(offset + 10, len(full_recommendations))
        new_recommendations = full_recommendations.iloc[offset:end_offset]
        st.session_state.recommendation_offsets[section_name] = offset + 10
    else:
        new_recommendations = data.sample(10, replace=True)

    st.session_state.sections[section_name] = new_recommendations


# -------------------------------
# Generate Recommendations Based on Type of Filtering User Selects
# -------------------------------

def generate_recommendations(rec_type, data, user_interactions):
    import pandas as pd

    # Normalize interactions to include watch_percent (None if missing)
    def normalize_interactions(interactions):
        return [list(i) if len(i) == 4 else list(i) + [None] for i in interactions]


    user_interactions_df = pd.DataFrame(
        normalize_interactions(user_interactions),
        columns=["title", "category", "interaction", "watch_percent"]
    )

    # Get liked titles
    liked_titles = user_interactions_df[
        user_interactions_df["interaction"] == "liked"
    ]["title"].tolist()

    # Extract watched titles + counts (fallback if needed)
    watched_titles = {
        row["title"]: row["watch_percent"] or 0
        for _, row in user_interactions_df.iterrows()
        if row["interaction"] == "watched"
    }

    # Collaborative Filtering
    if rec_type == "Recommendations based on what people like me watch":
        recommended_titles = get_similar_items(
            liked_titles, watched_titles, similarity_model, n=10
        )
        recommendations = data[data["title"].isin(recommended_titles)]

    # Content-Based Filtering
    elif rec_type == "Recommendations based on content I like":
        recommendations = get_content_based_recommendations(
            data, {"user_interactions": user_interactions}
        )

    # Hybrid: Both
    elif rec_type == "Both":
        collab_recommended_titles = get_similar_items(
            liked_titles, watched_titles, similarity_model, n=10
        )
        collab_recommendations = data[data["title"].isin(collab_recommended_titles)]

        content_recommendations = get_content_based_recommendations(
            data, {"user_interactions": user_interactions}
        )

        # Combine and shuffle
        recommendations = pd.concat(
            [collab_recommendations, content_recommendations]
        ).drop_duplicates()
        recommendations = recommendations.sample(frac=1).reset_index(drop=True)

    else:
        recommendations = data.sample(10, replace=True)

    # Add fallback relevance score
    if "relevance_score" not in recommendations.columns:
        recommendations["relevance_score"] = 1.0

    # Apply fairness re-ranking
    return get_fair_recommendations(recommendations)


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
                available_items = data[data["category"] == current_category].drop_duplicates(subset="title")
                st.session_state.current_items = available_items.sample(min(3, len(available_items)), replace=False)

            cols = st.columns(3)
            
            for i, (_, row) in enumerate(st.session_state.current_items.iterrows()):
                with cols[i]:
                    st.image(row["image"])
                    st.write(f"**{row['title']}**")
                    st.write(row["description"])
                    
                    like_key = f"like_{i}_{row['title']}"
                    dislike_key = f"dislike_{i}_{row['title']}"
                    
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

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🔽 Choose Recommendation Type", expanded=True):
        rec_type = st.selectbox(
            "Choose based on what you would like to receive personalized recommendations:",
            options=[
                "Recommendations based on what people like me watch",
                "Recommendations based on content I like",
                "Both"
            ],
            index=0,
            key="recommendation_type"
        )

    # Initialize session state if not already initialized
    if "current_rec_type" not in st.session_state:
        st.session_state.current_rec_type = rec_type

    # Check is session is updated
    if "current_rec_type" not in st.session_state or st.session_state.current_rec_type != rec_type:
        st.session_state.current_rec_type = rec_type

        # Always update recommendations
        filtered_recommendations = generate_recommendations(st.session_state.current_rec_type, data, st.session_state.user_interactions)

        # Update session_state with the new recommendations
        st.session_state.sections = {
            "For You": filtered_recommendations.head(10),
            "Interested": get_interest_based_recommendations(filtered_recommendations).head(10),
            "Trending Now": data[data["title"].isin(top_trending_titles)].head(10),
            "Most Watched": data[data["title"].isin(top_watched_titles)].head(10),
            "Random Selection": data.sample(10, replace=True),
            "Diversity Spotlight": diverse_data.sample(10, replace=False)
        }

    if st.session_state.current_rec_type == "Recommendations based on what people like me watch":
        filtered_recommendations = generate_recommendations(st.session_state.current_rec_type, data, st.session_state.user_interactions)
        refresh_section("For You")  
        refresh_section("Interested")  

    if "sections" not in st.session_state:
        refresh_section("For You")
        refresh_section("Interested")

   
    for section_name, content in st.session_state.sections.items():
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
        
        # Iterate through content in chunks of 5 for better alignment
        content_list = list(content.iterrows()) 
        for row_start in range(0, len(content_list), 5):
            cols = st.columns(5)  
            
            # Populate each column with a recommendation (up to 5 per row)
            for i in range(5):
                if row_start + i < len(content_list):  
                    _, row = content_list[row_start + i]
                    with cols[i]:  
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
                        
                        button_key = f"view_{section_name}_{row_start + i}"
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

        # SHOWING WHY IT WAS RECOMMENDED
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
        # Watch percentage slider
        st.subheader("🎬 How much did you watch?")
        title = st.session_state.selected_broadcast["title"]
        slider_key = f"slider_{title}"  # unique per broadcast

        watched_percent = st.slider(
            "How much of this content have you watched:",
            min_value=0,
            max_value=100,
            value=0,
            key=slider_key,
        )


        title = st.session_state.selected_broadcast["title"]
        category = st.session_state.selected_broadcast["category"]

        # Remove previous "watched" or "skipped" entry if exists
        st.session_state.user_interactions = [
            i for i in st.session_state.user_interactions 
            if not (i[0] == title and i[2] in ["watched", "skipped"])
        ]

        # Determine action + weight
        if watched_percent == 0:
            action = "skipped"
            weight = -0.5
        else:
            action = "watched"
            if watched_percent == 100:
                weight = 0.75
            elif watched_percent >= 50:
                weight = 0.5
            else:
                weight = -0.25

        # Add the new interaction
        st.session_state.user_interactions.append((title, category, action, watched_percent))

        # Update watch count
        st.session_state.watch_counts[title] = st.session_state.watch_counts.get(title, 0) + 1

        # Update preferences based on interaction weight
        st.session_state.user_preferences[category] = st.session_state.user_preferences.get(category, 0) + weight

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

                if st.button("View", key=f"view_more_reco_{i}", on_click=view_broadcast, args=(row.to_dict(),)):
                    st.rerun()

        if st.button("🏠 Back to Home"):
            st.session_state.selected_broadcast = None
            st.rerun()

    except Exception as e:
        st.error(f"Something went wrong: {e}")
