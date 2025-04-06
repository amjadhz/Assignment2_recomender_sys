import streamlit as st
import pandas as pd
import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

# -------------------------------
# Lay-out
# -------------------------------
st.set_page_config(layout="wide", page_title="BBC Recommender System")

st.markdown(
    """
    <style>
        /* Widen the main container and soften spacing */
        .main .block-container {
            max-width: 1800px;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-top: 1rem;
        }

        /* Shrink top padding for a tighter UI */
        header[data-testid="stHeader"] {
            margin-bottom: 0.5rem;
        }

        /* Make images sleek with rounded corners and shadows */
        img {
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            object-fit: cover;
        }

        /* Headings - bold and spaced out */
        h1, h2, h3 {
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }

        /* Button spacing uniformity */
        button[kind="primary"] {
            margin-top: 0.5rem;
        }

        /* Tighter layout on recommendation tiles */
        .element-container {
            padding: 0.3rem !important;
        }

        /* Make selectboxes and buttons look clean */
        .stSelectbox > div {
            border-radius: 10px;
        }

        /* Divider styling */
        hr {
            border-top: 1px solid #eaeaea;
            margin-top: 2rem;
            margin-bottom: 2rem;
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

# Clear user data ONLY once at start of fresh session
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

def generate_interest_based_recommendations(data, user_interactions, min_items=10, overgenerate=40):
    # Step 1: Get user-preferred genres
    liked_categories = {
        inter[1] for inter in user_interactions
        if len(inter) > 2 and inter[2] == "liked"
    }

    # Step 2: Get more than needed collaborative recommendations
    def normalize_interactions(interactions):
        return [list(i) if len(i) == 4 else list(i) + [None] for i in interactions]

    df = pd.DataFrame(
        normalize_interactions(user_interactions),
        columns=["title", "category", "interaction", "watch_percent"]
    )

    liked_titles = df[df["interaction"] == "liked"]["title"].tolist()
    watched_titles = {
        row["title"]: row["watch_percent"] or 0
        for _, row in df.iterrows() if row["interaction"] == "watched"
    }

    similar_titles = get_similar_items(liked_titles, watched_titles, similarity_model, n=overgenerate)
    collab_df = data[data["title"].isin(similar_titles)]

    # Step 3: Filter by liked genres
    interest_recommendations = collab_df[collab_df["category"].isin(liked_categories)]

    # Step 4: Fill with content-based if fewer than needed
    if len(interest_recommendations) < min_items:
        needed = min_items - len(interest_recommendations)
        filler = get_content_based_recommendations(data, {"user_interactions": user_interactions})
        filler = filler[filler["category"].isin(liked_categories)]
        filler = filler[~filler["title"].isin(interest_recommendations["title"])]
        interest_recommendations = pd.concat([interest_recommendations, filler]).drop_duplicates(subset="title")

    # Step 5: Fairness re-ranking
    if "relevance_score" not in interest_recommendations.columns:
        interest_recommendations["relevance_score"] = 1.0
    interest_recommendations = get_fair_recommendations(interest_recommendations)

    return interest_recommendations.head(min_items)



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

import random

def get_similar_items(user_likes, user_watches, model, n=10, overgenerate=40):
    if model.empty or (not user_likes and not user_watches):
        return []  # Return empty list if no data

    similarity_scores = pd.Series(0.0, index=model.columns)

    # Add scores for liked items
    for title in user_likes:
        if title in model.index:
            similarity_scores = similarity_scores.add(model[title] * 1.0, fill_value=0)

    # Add scores for watched items
    for title, count in user_watches.items():
        if title in model.index:
            similarity_scores = similarity_scores.add(model[title] * (0.5 * count), fill_value=0)

    # Remove already interacted titles
    titles_to_remove = user_likes + list(user_watches.keys())
    similarity_scores = similarity_scores.drop(labels=titles_to_remove, errors="ignore")


    # If nothing left
    if similarity_scores.empty:
        return []

    # Get top `overgenerate` similar items
    top_candidates = similarity_scores.sort_values(ascending=False).head(overgenerate).index.tolist()

    # Sample `n` unique items randomly from those top candidates
    return random.sample(top_candidates, min(n, len(top_candidates)))


# -------------------------------
# Trending Items Functions
# -------------------------------

@st.cache_data
def load_synthetic_interactions(path="data/synthetic_user_interactions.csv"):
    try:
        df = pd.read_csv(path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Failed to load synthetic interactions: {e}")
        return pd.DataFrame()

# Function to compute trending titles based on recent watch behavior
def get_recent_trending_titles(df, days=7):
    cutoff = datetime.now() - timedelta(days=days)
    recent_df = df[(df["Watch %"] > 0) & (df["Timestamp"] >= cutoff)]
    title_counts = recent_df["Title"].value_counts()
    return title_counts.head(20).index.tolist()

# Load the data once
synthetic_df = load_synthetic_interactions()
top_trending_titles = get_recent_trending_titles(synthetic_df)

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
    # Normalize and extract interactions
    if isinstance(user_interactions, dict) and "user_interactions" in user_interactions:
        interactions_list = user_interactions["user_interactions"]
    else:
        interactions_list = user_interactions

    # Ensure uniform 4-column format
    def normalize_interactions(interactions):
        return [list(i) if len(i) == 4 else list(i) + [None] for i in interactions]

    normalized = normalize_interactions(interactions_list)

    user_interactions_df = pd.DataFrame(
        normalized,
        columns=["title", "category", "interaction", "watch_percent"]
    )

    # Get liked, disliked, and skipped items
    liked_items = user_interactions_df[user_interactions_df["interaction"] == "liked"]["title"].tolist()
    disliked_items = user_interactions_df[user_interactions_df["interaction"] == "disliked"]["title"].tolist()
    skipped_items = user_interactions_df[user_interactions_df["interaction"] == "skipped"]["title"].tolist()

    # Get watched items and their watch counts
    watch_counts = {
        row["title"]: row["watch_percent"] or 0
        for _, row in user_interactions_df.iterrows()
        if row["interaction"] == "watched"
    }

    # Build similarity matrix using the description column
    similarity_matrix = build_similarity_matrix(data)

    recommended_items = set()
    excluded_items = set(disliked_items + skipped_items)

    # Exclude items similar to disliked ones
    for title in disliked_items:
        if title in data['title'].values:
            item_index = data[data['title'] == title].index[0]
            similar_indices = similarity_matrix[item_index].argsort()[-11:-1]
            excluded_items.update(data.iloc[similar_indices]['title'].values)

    # Find similar items to liked titles
    for title in liked_items:
        if title in data['title'].values:
            item_index = data[data['title'] == title].index[0]
            similar_indices = similarity_matrix[item_index].argsort()[-11:-1]
            recommended_items.update(data.iloc[similar_indices]['title'].values)

    # Find similar items to watched titles
    for title in watch_counts.keys():
        if title in data['title'].values:
            item_index = data[data['title'] == title].index[0]
            similar_indices = similarity_matrix[item_index].argsort()[-6:-1]
            recommended_items.update(data.iloc[similar_indices]['title'].values)

    # Remove any disliked/skipped or similar-to-disliked items
    recommended_items -= excluded_items

    # Return empty DataFrame if nothing left
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

    # Init sections if missing
    if 'sections' not in st.session_state:
        st.session_state.sections = {
            "For You": filtered_recommendations.head(10),
            "Based on genres you like!": filtered_recommendations.head(10),
            "Trending Now": data[data["title"].isin(top_trending_titles)].head(10),
            "Most Watched": data[data["title"].isin(top_watched_titles)].head(10),
            "Random Selection": data.sample(10, replace=True),
            "Diversity Spotlight": diverse_data.sample(10, replace=False)
        }

    # Init offsets
    if 'recommendation_offsets' not in st.session_state:
        st.session_state.recommendation_offsets = {
            "For You": 0,
            "Based on genres you like!": 0
        }

    # Log interaction as skipped for items in this section
    try:
        for _, row in st.session_state.sections[section_name].iterrows():
            st.session_state.user_interactions.append((row['title'], row['category'], "skipped"))
    except:
        pass

    save_user_data()

    # Handle dynamic sections
    if section_name == "For You":
        full_recommendations = generate_recommendations(
            current_rec_type, data, st.session_state.user_interactions
        ).drop_duplicates()
    elif section_name == "Based on genres you like!":
        # Always refresh the genre-based pool (instead of caching it)
        st.session_state.genre_rec_pool = generate_interest_based_recommendations(
            data, st.session_state.user_interactions, min_items=30
        )
        full_recommendations = st.session_state.genre_rec_pool

    elif section_name == "Diversity Spotlight":
        new_recommendations = diverse_data.sample(10, replace=False)
        st.session_state.sections[section_name] = new_recommendations
        return
    else:
        new_recommendations = data.sample(10, replace=True)
        st.session_state.sections[section_name] = new_recommendations
        return

    # Paging logic
    offset = st.session_state.recommendation_offsets.get(section_name, 0)
    if offset + 10 >= len(full_recommendations):
        offset = 0
    end_offset = min(offset + 10, len(full_recommendations))
    new_recommendations = full_recommendations.iloc[offset:end_offset].copy()
    st.session_state.recommendation_offsets[section_name] = offset + 10

    # Save result
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
            "Based on genres you like!": generate_interest_based_recommendations(data, st.session_state.user_interactions),
            "Trending Now": data[data["title"].isin(top_trending_titles)].head(10),
            "Most Watched": data[data["title"].isin(top_watched_titles)].head(10),
            "Random Selection": data.sample(10, replace=True),
            "Diversity Spotlight": diverse_data.sample(10, replace=False)
        }

    # Always update before refreshing
    filtered_recommendations = generate_recommendations(
        st.session_state.current_rec_type, data, st.session_state.user_interactions
    )

    if st.session_state.current_rec_type == "Recommendations based on what people like me watch":
        refresh_section("For You")  
        refresh_section("Based on genres you like!")


   
    for section_name, content in st.session_state.sections.items():
        # ⚠️ Show warning if too few results in the genre section
        if section_name == "Based on genres you like!" and len(content) < 3:
            st.warning("Not enough items found for your liked genres — try exploring more content!")

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
