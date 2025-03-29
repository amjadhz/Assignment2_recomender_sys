import pandas as pd
import re
from collections import Counter
from rapidfuzz import fuzz

def simplify_title(title):
    # Remove common episode indicators and keep core part
    title = re.sub(r"( - )?(Series|Episode|Ep|Part|Season|S\d+:E\d+|\d{1,2}\.\s).*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[\[\]\(\)]", "", title)  # Remove brackets
    return title.strip()

def load_and_group_shows(file_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Drop rows missing important data
    df.dropna(subset=["title", "description", "image"], inplace=True)

    # Simplify titles for grouping
    df["base_title"] = df["title"].apply(simplify_title)

    # Count frequency of each base title
    title_counts = df["base_title"].value_counts()

    # Pick top 1 entry per base title (most common full title under that base)
    grouped_rows = []
    for base_title, count in title_counts.items():
        group = df[df["base_title"] == base_title]
        best_row = group.iloc[0]  # could also do: group.sort_values("title").iloc[0]
        best_row["title"] = base_title  # replace full title with simplified title
        grouped_rows.append(best_row)

    df_cleaned = pd.DataFrame(grouped_rows)
    df_cleaned.drop(columns=["base_title"], inplace=True)
    df_cleaned.reset_index(drop=True, inplace=True)
    df_cleaned.to_csv(output_path, index=False)

    print(f"✅ Grouped and cleaned dataset saved to '{output_path}' with {len(df_cleaned)} shows.")
    return df_cleaned

# Example usage
load_and_group_shows(
    file_path="data/bbc_recommender_dataset_clean.csv",
    output_path="data/bbc_recommender_dataset_clean.csv"
)
