import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def random_timestamp(days_back=365):
    base_date = datetime.now() - timedelta(days=random.randint(0, days_back))
    return base_date.strftime("%Y-%m-%d %H:%M:%S")

def synthesize_user_data(path_to_dataset="data/bbc_recommender_dataset.csv", n_users=100):
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
            num_interactions = random.randint(3, 7)

            for _ in range(num_interactions):
                title = random.choice(real_titles)
                category = bbc_data.loc[bbc_data["title"] == title, "category"].values[0]

                # Weight selection to bias based on persona
                weight = preferences.get(category, 0.01)
                if random.random() < weight:
                    rating = random.randint(4, 5)
                else:
                    rating = random.randint(1, 3)

                interaction = {
                    "User ID": f"user_{user_id}",
                    "Persona": persona,
                    "Title": title,
                    "Category": category,
                    "Rating": rating,
                    "Watched Completely": "Yes" if rating >= 4 else "No",
                    "Device": random.choice(devices),
                    "Timestamp": random_timestamp()
                }

                synthetic_users.append(interaction)

        return pd.DataFrame(synthetic_users)

    except FileNotFoundError:
        print("Error: Dataset not found.")
        return None

if __name__ == "__main__":
    df = synthesize_user_data()
    df.to_csv("data/synthetic_user_interactions.csv", index=False)
    print("Synthetic user interaction data saved successfully!")