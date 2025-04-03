import os
import json
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def load_user_jsons(folder_path):
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            user_id = filename.replace(".json", "")
            with open(os.path.join(folder_path, filename), "r") as f:
                user_data = json.load(f)
                for interaction in user_data["user_interactions"]:
                    title, category = interaction[0], interaction[1]
                    action = interaction[2]
                    watch_percent = interaction[3] if len(interaction) > 3 else None

                    # Assign rating based on action and watch %
                    if action == "liked":
                        rating = 5
                    elif action == "disliked":
                        rating = 1
                    elif action == "skipped":
                        rating = 2
                    elif action == "watched" and watch_percent is not None:
                        if watch_percent >= 100:
                            rating = 4
                        elif watch_percent >= 50:
                            rating = 3
                        else:
                            rating = 2
                    else:
                        rating = 3  # Neutral fallback

                    data.append((user_id, title, rating))

    return pd.DataFrame(data, columns=["user_id", "title", "rating"])


def build_item_similarity(df):
    # Create user-item rating matrix
    matrix = df.pivot_table(index="user_id", columns="title", values="rating").fillna(0)

    # Compute cosine similarity between items (transpose for item-based)
    similarity = cosine_similarity(matrix.T)

    # Wrap it in a DataFrame for easy use
    sim_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
    return sim_df

if __name__ == "__main__":
    print("Loading user data...")
    df = load_user_jsons("data/user_json_profiles")

    print("Building item-item similarity matrix...")
    item_similarity_df = build_item_similarity(df)

    # Save model (just the similarity matrix for now)
    os.makedirs("models", exist_ok=True)
    with open("models/item_similarity.pkl", "wb") as f:
        pickle.dump(item_similarity_df, f)

    print("🎉 Model saved to models/item_similarity.pkl")

