# Assignment2_recomender_sys
Repository can be found in - [link](https://github.com/amjadhz/Assignment2_recomender_sys) 
## ✍️ Authors

- Amjad Hwidy – [GitHub](https://github.com/amjadhz)
- David Verboom – [GitHub](https://github.com/davidverboom)
- Lena Verwilghen – [GitHub](https://github.com/Verwilghen)
- Anna Grabowska – [GitHub](https://github.com/Ania2901)

This repository contains the code for creating a recommender system as part of Assignment 2 in the Personalization course.

## 1. Setting Up the Environment

### Create a Virtual Environment

To ensure dependencies are installed in an isolated environment, create a virtual environment using Python:

```sh
python -m venv .venv
```

### Activate the Virtual Environment

- **On Windows** (Command Prompt):
  ```sh
  .venv\Scripts\activate
  ```
  
- **On macOS/Linux**:
  ```sh
  source .venv/bin/activate
  ```

### Install Required Libraries

Once the virtual environment is activated, install the required dependencies from `requirements.txt`:

```sh
pip install -r requirements.txt
```

### Verify Installation

To confirm that all dependencies are installed correctly, run:

```sh
pip list
```

You are now ready to run the recommender system!

## Running the Recommender System

To execute the main script, run:

```sh
python app.py
```

Ensure that all data files are available in the expected paths before running the system.

## Deactivating the Virtual Environment

Once you are done, deactivate the virtual environment by running:

```sh
deactivate
```



## User App
In this application you can see how the app will looklike for user

To run it use command 

```sh
streamlit run "user_app.py"
```

## Developer Dashboard 

This is another app can be run in parallel with the user app to see the user interactions.

Here where we put our metrics for the values we choose 

To run it use command 

```sh
streamlit run "developer_dashboard.py"
```

## 2. Data Cleaning Process

To ensure high-quality input for the recommender system, a data preprocessing script was developed to clean and consolidate the raw dataset. The steps below outline the data cleaning methodology applied:

### 1. Removal of Incomplete Records

Entries missing any of the following critical fields were excluded from the dataset:
- `title`
- `description`
- `image`

This step ensured that all remaining records contained the necessary metadata for display and recommendation purposes.

### 2. Title Simplification

Broadcast titles often contained extraneous episode-level information such as:
- Episode numbers (e.g., `"Episode 3"`, `"Ep. 5"`)
- Series or season references (e.g., `"Season 2"`, `"S2:E1"`)
- Numbered prefixes (e.g., `"1. Introduction"`)

These patterns were systematically removed using regular expressions, resulting in cleaner, more consistent base titles.  

### 3. Grouping of Similar Titles

After simplification, titles were grouped by their cleaned base form. Within each group, a representative entry was selected (typically the first occurrence), and duplicates were discarded. This reduced redundancy in the content catalog while preserving the uniqueness of each show or series.

### 4. Exporting the Cleaned Dataset

The cleaned dataset was saved to a new CSV file and used throughout the application for training, recommendations, and evaluation.


markdown
Kopiëren
Bewerken
## 3. Synthetic Data Generation

To simulate realistic user behavior for experimentation and evaluation purposes, synthetic user interaction data was generated using a custom Python script.

This process produced two key datasets:
- `synthetic_user_interactions.csv` – a flat log of timestamped user viewing behavior
- Individual JSON user profiles in `/data/user_json_profiles/` – used by the recommender system for personalization

### 1. User Persona Modeling

Five distinct **user personas** were defined to emulate varied viewing habits:
- **Documentary Lover**
- **Sports Fan**
- **News Watcher**
- **Entertainment Junkie**
- **Balanced Viewer**

Each persona had different preferences across content categories, influencing what they were likely to watch and how much of it they would consume.

### 2. Interaction Simulation

For each synthetic user:
- A random persona was assigned.
- 25–30 interactions were generated based on their category preferences.
- Each interaction included:
  - **Title**
  - **Category**
  - **Device used** (e.g., TV, Smartphone)
  - **Watch percentage** (e.g., 100%, 75%, 0%, etc.)
  - **Timestamp** (randomly assigned within the past year)

Watch percentages were sampled using weighted probabilities based on the user’s interest in the content category (e.g., high interest → more likely to watch fully).

### 3. Deriving User Preferences and Behaviors

Using the watch percentages, each interaction was translated into one of the following behaviors:
- `liked`
- `watched`
- `skipped`
- `disliked` (5% chance at random)

These behaviors were used to:
- Accumulate **category-level preferences** (from -1.5 to 1.5)
- Count **watch frequency per title**
- Create a history of interactions for each user

### 4. Output Format

Each user was saved as a structured JSON file containing:
- `user_preferences`: a mapping of category to preference weight
- `user_interactions`: a list of individual content interactions
- `watch_counts`: how many times each title was viewed

## 4. Collaborative Filtering Model

To generate recommendations based on user behavior, an **item-based collaborative filtering** model was developed. This approach relies on analyzing the interactions of synthetic users with different shows, identifying patterns of co-engagement to surface relevant recommendations.

### Overview of the Process

The collaborative model construction consists of three primary steps:

---

### 1. Loading and Preprocessing User Interaction Data

Each synthetic user is represented as a `.json` profile containing:
- A list of interactions (e.g., `liked`, `watched`, `skipped`, `disliked`)
- Optional watch percentage (e.g., `80%`)

These interactions are converted into a numerical **rating system** using the following mapping:

| Interaction Type | Watch % (if applicable) | Assigned Rating |
|------------------|-------------------------|-----------------|
| `liked`          | N/A                     | `5`             |
| `disliked`       | N/A                     | `1`             |
| `skipped`        | N/A                     | `2`             |
| `watched`        | `≥ 100%`                | `4`             |
| `watched`        | `≥ 50%`                 | `3`             |
| `watched`        | `< 50%`                 | `2`             |
| _Fallback_       | —                       | `3` (neutral)   |

The final dataset is a list of `(user_id, title, rating)` records.

---

### 2. Building the User-Item Matrix

The interaction data is pivoted into a **user-item matrix**:

```plaintext
Rows   → users  
Columns → titles  
Values → ratings
```

## Recommendation Model: Collaborative Filtering

The recommender system uses **item-based collaborative filtering** to suggest content based on user behavior patterns.

### Model Workflow

1. **Input Data**  
   The model uses synthetic user profiles stored in `/data/user_json_profiles/`, which contain detailed user interactions with various programs (e.g., liked, watched, skipped).

2. **Rating Conversion**  
   Interactions are converted into a numerical rating scale (1–5) based on both the type of interaction and the percentage of content watched.

3. **User-Item Matrix**  
   A matrix is built with users as rows, titles as columns, and ratings as values. Missing interactions are filled with zero.

4. **Similarity Computation**  
   Using cosine similarity, the model computes how similar each item (TV show or program) is to every other item, based on user co-engagement.

## How the App Works

The `user_app.py` is an interactive Streamlit interface where users can:

1. **Select their favorite genres** (minimum 5).
2. **Rate a few handpicked examples** from each liked genre (like/dislike).
3. **Receive personalized recommendations** across several sections:
   - **For You** – General recommendations based on your behavior and others like you.
   - **Based on genres you like!** – Focused on your selected categories.
   - **Trending Now** – Based on recent synthetic user trends.
   - **Most Watched** – Frequently viewed across all users.
   - **Random Selection** – For serendipitous discovery.
   - **Diversity Spotlight** – Ensures a variety of topics.

Users can:
- View content, like/dislike it, and indicate how much they watched.
- Get explanations on *why* something was recommended.
- Refresh each section to explore more titles.

Behind the scenes, the app uses:
- **Collaborative filtering** (via an item-item similarity model)
- **Content-based filtering** (TF-IDF on descriptions)
- **A fairness-adjusted re-ranking** to avoid over-recommending the same items.



  

