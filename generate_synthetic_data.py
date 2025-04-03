import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

def random_timestamp(days_back=365):
    base_date = datetime.now() - timedelta(days=random.randint(0, days_back))
    return base_date.strftime("%Y-%m-%d %H:%M:%S")


def synthesize_user_data(path_to_dataset="data/bbc_recommender_dataset_clean.csv", n_users=100):
    try:
        bbc_data = pd.read_csv(path_to_dataset)

        real_titles = bbc_data["title"].dropna().unique().tolist()
        real_categories = bbc_data["category"].dropna().unique().tolist()

        # Define user personas based on category preferences
        bbc_personas = {
            "Documentary Lover": {cat: 0.9 if cat == "Documentaries" else 0.05 for cat in real_categories},
            "Sports Fan": {cat: 0.9 if cat == "Sport" else 0.05 for cat in real_categories},
            "News Watcher": {cat: 0.9 if cat == "News" else 0.05 for cat in real_categories},
            "Entertainment Junkie": {cat: 0.9 if cat == "Entertainment" else 0.05 for cat in real_categories},
            "Balanced Viewer": {cat: 1 / len(real_categories) for cat in real_categories}
        }

        synthetic_users = []
        devices = ["TV", "Smartphone", "Laptop", "Tablet"]

        for user_id in range(1, n_users + 1):
            persona = random.choice(list(bbc_personas.keys()))
            preferences = bbc_personas[persona]
            num_interactions = random.randint(25, 35)

            for _ in range(num_interactions):
                title = random.choice(real_titles)
                category = bbc_data.loc[bbc_data["title"] == title, "category"].values[0]
                interest = preferences.get(category, 0.05)

                # Realistic watch % distribution
                if interest > 0.8:
                    watch_percent = random.choices(
                        [100, 95, 90, 75, 50, 30],
                        weights=[0.5, 0.2, 0.15, 0.1, 0.03, 0.02]
                    )[0]
                elif interest > 0.4:
                    watch_percent = random.choices(
                        [100, 90, 75, 50, 25, 10],
                        weights=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
                    )[0]
                else:
                    watch_percent = random.choices(
                        [100, 75, 50, 25, 10, 0],
                        weights=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1]
                    )[0]

                synthetic_users.append({
                    "User ID": f"user_{user_id}",
                    "Persona": persona,
                    "Title": title,
                    "Category": category,
                    "Device": random.choice(devices),
                    "Timestamp": random_timestamp(),
                    "Watch %": watch_percent
                })

        return pd.DataFrame(synthetic_users)

    except FileNotFoundError:
        print("❌ Error: Dataset not found.")
        return None


def build_user_json_profiles(df, output_dir="data/user_json_profiles"):
    os.makedirs(output_dir, exist_ok=True)
    user_profiles = {}

    for user_id, user_data in df.groupby("User ID"):
        user_preferences = defaultdict(float)
        user_interactions = []
        watch_counts = defaultdict(int)

        for _, row in user_data.iterrows():
            title = row["Title"]
            category = row["Category"]
            watch_percent = row["Watch %"]

            # 5% chance to dislike anything
            if random.random() < 0.05:
                action = "disliked"
                weight = -1.0
                user_interactions.append([title, category, action])
                user_preferences[category] += weight
                continue

            # Interpret watch % into behavior
            if watch_percent == 0:
                action = "skipped"
                weight = -0.5
                user_interactions.append([title, category, action])
            elif watch_percent == 100:
                action = "liked"
                weight = 1.0
                user_interactions.append([title, category, action])
            elif watch_percent >= 85:
                # 70% chance to like if almost fully watched
                if random.random() < 0.7:
                    action = "liked"
                    weight = 1.0
                    user_interactions.append([title, category, action])
                else:
                    action = "watched"
                    weight = 0.5
                    user_interactions.append([title, category, action, watch_percent])
            elif watch_percent >= 50:
                action = "watched"
                weight = 0.5
                user_interactions.append([title, category, action, watch_percent])
            elif watch_percent >= 25:
                action = "watched"
                weight = 0.0
                user_interactions.append([title, category, action, watch_percent])
            else:
                action = "watched"
                weight = -0.25
                user_interactions.append([title, category, action, watch_percent])

            user_preferences[category] += weight
            if watch_percent > 0:
                watch_counts[title] += 1

        # Clamp preferences to a realistic range
        for cat in user_preferences:
            user_preferences[cat] = max(-1.5, min(user_preferences[cat], 1.5))

        user_profile = {
            "user_preferences": dict(user_preferences),
            "user_interactions": user_interactions,
            "watch_counts": dict(watch_counts)
        }

        with open(os.path.join(output_dir, f"{user_id}.json"), "w") as f:
            json.dump(user_profile, f, indent=2)

        user_profiles[user_id] = user_profile

        # Optional debug
        avg_score = np.mean(list(user_preferences.values()))
        liked_count = sum(1 for i in user_interactions if i[2] == "liked")
        print(f"✅ {user_id} | Likes: {liked_count} | Avg sentiment: {round(avg_score,2)}")

    print(f"🎉 Saved {len(user_profiles)} user profiles in '{output_dir}'")
    return user_profiles


if __name__ == "__main__":
    df = synthesize_user_data()
    if df is not None:
        df.to_csv("data/synthetic_user_interactions.csv", index=False)
        build_user_json_profiles(df)
