import streamlit as st
import pandas as pd
import json

# Function to clear user interactions file
def clear_user_data():
    with open("./data/user_interactions.json", "w") as f:
        json.dump({"user_preferences": {}, "user_interactions": []}, f)

# Clear JSON file at the start of the app
clear_user_data()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("./data/bbc_recommender_dataset.csv")

data = load_data()

# Function to save user interactions to a file
def save_user_data():
    with open("./data/user_interactions.json", "w") as f:
        json.dump({
            "user_preferences": st.session_state.user_preferences,
            "user_interactions": st.session_state.user_interactions
        }, f)

# Load previous interactions
def load_user_data():
    try:
        with open("./data/user_interactions.json", "r") as f:
            data = json.load(f)
            return data.get("user_preferences", {}), data.get("user_interactions", [])
    except FileNotFoundError:
        return {}, []

# Initialize session states
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

# Step 1: User Chooses Preferences
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

# Home Page after choosing preferences
elif st.session_state.selected_broadcast is None:
    st.title("BBC Recommender System - User Interface")
    st.header("Your Recommendations")
    
    sections = {
        "For You": data.sample(10, replace=True),
        "Last Watched": data.sample(10, replace=True),
        "Might Like to Watch": data.sample(10, replace=True),
        "Trending or Popular": data.sample(10, replace=True)
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
    
    st.subheader("More Recommendations")
    more_recommendations = data.sample(10, replace=True)
    cols = st.columns(5)
    for i, (_, row) in enumerate(more_recommendations.iterrows()):
        with cols[i % 5]:
            st.image(row["image"],)
            st.write(f"**{row['title']}**")
            if st.button(f"View {row['title']}", key=f"view_more_{row['title']}"):
                st.session_state.selected_broadcast = row.to_dict()
                st.rerun()
    
    if st.button("Back to Home"):
        st.session_state.selected_broadcast = None
        st.rerun()

# Run the user app with: streamlit run user_app.py
