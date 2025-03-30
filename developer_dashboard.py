import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------
# File Paths
# -------------------------------
USER_DATA_FILE = "./data/user_interactions.json"
SYNTHETIC_DIR = "./data/user_json_profiles"
DATASET_FILE = "./data/bbc_recommender_dataset.csv"

# -------------------------------
# Load BBC Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATASET_FILE)

# -------------------------------
# Load Live User Data
# -------------------------------
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {"user_preferences": {}, "user_interactions": [], "watch_counts": {}}
    return data.get("user_preferences", {}), data.get("user_interactions", []), data.get("watch_counts", {})

# -------------------------------
# Load Synthetic User Data
# -------------------------------
def load_synthetic_user_data():
    prefs, interactions, watches = {}, [], {}
    for filename in os.listdir(SYNTHETIC_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(SYNTHETIC_DIR, filename), "r") as f:
                user_data = json.load(f)
                for cat, val in user_data.get("user_preferences", {}).items():
                    prefs[cat] = prefs.get(cat, 0) + val
                interactions.extend(user_data.get("user_interactions", []))
                for title, count in user_data.get("watch_counts", {}).items():
                    watches[title] = watches.get(title, 0) + count
    return prefs, interactions, watches

# -------------------------------
# Initialize Data
# -------------------------------
data = load_data()
live_prefs, live_interactions, live_watch_counts = load_user_data()
syn_prefs, syn_interactions, syn_watch_counts = load_synthetic_user_data()

# Combined Data
overall_prefs = live_prefs.copy()
for cat, val in syn_prefs.items():
    overall_prefs[cat] = overall_prefs.get(cat, 0) + val

overall_interactions = live_interactions + syn_interactions

overall_watch_counts = live_watch_counts.copy()
for title, count in syn_watch_counts.items():
    overall_watch_counts[title] = overall_watch_counts.get(title, 0) + count

# Store in session state
st.session_state.user_preferences = live_prefs
st.session_state.user_interactions = live_interactions
st.session_state.watch_counts = live_watch_counts
st.session_state.synthetic_interactions = syn_interactions

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="BBC Recommender Dashboard")
st.title("📊 BBC Recommender System - Developer Dashboard")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, autonomy_tab, transparency_tab, diversity_tab = st.tabs([
    "📈 Overall", "⚖️ Fairness Analysis", "🧭 Autonomy", "🔍 Transparency", "🌈 Diversity"
])

with tab1:
    # -------------------------------
    # Live User Category Summary
    # -------------------------------
    st.subheader("🧑 Live User Summary")

    preferred_categories = [cat for cat, score in live_prefs.items() if score > 0]
    if preferred_categories:
        st.subheader("🔥Preferred Categories: ")
        # st.success("🔥Preferred Categories: ")
        st.subheader("" + ", ".join(preferred_categories))
    else:
        st.info("No preferred categories found for the live user.")

    # -------------------------------
    # Metrics
    # -------------------------------
    st.subheader("📊 User Metrics")

    pref_likes = sum(1 for v in live_prefs.values() if v > 0)
    pref_dislikes = sum(1 for v in live_prefs.values() if v < 0)

    interaction_likes = sum(1 for i in live_interactions if len(i) > 2 and i[2] == "liked")
    interaction_dislikes = sum(1 for i in live_interactions if len(i) > 2 and i[2] == "disliked")

    live_likes = interaction_likes
    live_dislikes = interaction_dislikes
    live_watched = sum(live_watch_counts.values())

    overall_likes = pref_likes + interaction_likes
    overall_dislikes = pref_dislikes + interaction_dislikes

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👍 Live Likes", live_likes, help=f"{interaction_likes} from interactions, {pref_likes} from preferences")
        st.metric("👍 Total Likes",  overall_likes, help=f"{interaction_likes} from interactions, {pref_likes} from preferences")
    with col2:
        st.metric("👎 Live Dislikes", live_dislikes)
        st.metric("👎 Total Dislikes", overall_dislikes, help=f"{interaction_dislikes} from interactions, {pref_dislikes} from preferences")
    with col3:
        st.metric("👁️ Watched (Live)", live_watched)

    # -------------------------------
    # Preferences Table
    # -------------------------------
    st.subheader("📂 Live Preferences")
    if live_prefs:
        st.dataframe(pd.DataFrame(live_prefs.items(), columns=["Category", "Score"]))
    else:
        st.info("No preferences recorded.")

    # -------------------------------
    # Interaction Table
    # -------------------------------
    st.subheader("📋 Live Interactions")
    if live_interactions:
        st.dataframe(pd.DataFrame([
    i if len(i) == 3 else i[:3] + [i[3]] for i in live_interactions
], columns=["Title", "Category", "Action", "Extra"] if any(len(i) > 3 for i in live_interactions) else ["Title", "Category", "Action"]))
    else:
        st.info("No interactions recorded.")

    # -------------------------------
    # Likes and Dislikes Charts
    # -------------------------------
    if live_interactions:
        df_live = pd.DataFrame([
    i if len(i) == 3 else i[:3] + [i[3]] for i in live_interactions
], columns=["Title", "Category", "Action", "Extra"] if any(len(i) > 3 for i in live_interactions) else ["Title", "Category", "Action"])
        like_df = df_live[df_live["Action"] == "liked"]
        dislike_df = df_live[df_live["Action"] == "disliked"]

        st.subheader("👍 Likes per Category")
        if not like_df.empty:
            like_counts = like_df["Category"].value_counts().sort_index()
            fig, ax = plt.subplots()
            sns.barplot(x=like_counts.index, y=like_counts.values, color="green", ax=ax)
            ax.set_title("Liked Categories")
            ax.set_xlabel("Category")
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        else:
            st.info("No likes yet.")

        st.subheader("👎 Dislikes per Category")
        if not dislike_df.empty:
            dislike_counts = dislike_df["Category"].value_counts().sort_index()
            fig, ax = plt.subplots()
            sns.barplot(x=dislike_counts.index, y=dislike_counts.values, color="red", ax=ax)
            ax.set_title("Disliked Categories")
            ax.set_xlabel("Category")
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        else:
            st.info("No dislikes yet.")
    else:
        st.warning("No interactions yet to analyze.")

with tab2:
    st.header("⚖️ Fairness Analysis")

    st.markdown("""
    This section explores fairness in recommendations by analyzing whether the recommender system:
    - Exposes users to a balanced range of categories
    - Aligns recommendations with user preferences
    - Avoids over-favoring popular content
    """)

    # Prepare data
    if live_interactions:
        df_interactions = pd.DataFrame([
    i if len(i) == 3 else i[:3] + [i[3]] for i in live_interactions
], columns=["Title", "Category", "Action", "Extra"] if any(len(i) > 3 for i in live_interactions) else ["Title", "Category", "Action"])
    else:
        st.warning("No interactions available to analyze fairness.")
        st.stop()

    # -------------------------------
    # 1. Exposure per Category
    # -------------------------------
    st.subheader("1. Exposure Distribution per Category")

    exposure_counts = df_interactions["Category"].value_counts(normalize=True).sort_index()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=exposure_counts.index, y=exposure_counts.values, palette="viridis", ax=ax1)
    ax1.set_title("Relative Exposure of Categories (Live User)")
    ax1.set_ylabel("Proportion of Interactions")
    ax1.set_xlabel("Category")
    ax1.tick_params(axis="x", rotation=45)
    st.pyplot(fig1)

    max_cat = exposure_counts.idxmax()
    max_prop = exposure_counts.max()
    if max_prop > 0.5:
        st.warning(f"⚠️ Over {round(max_prop*100)}% of content is from '{max_cat}' — this suggests a potential **category bias**.")
    else:
        st.success("✅ The exposure appears fairly distributed across categories.")

    # -------------------------------
    # 2. Exposure vs Preference Alignment
    # -------------------------------
    st.subheader("2. Preference vs. Exposure Alignment")

    pref_df = pd.DataFrame(list(live_prefs.items()), columns=["Category", "Preference"])
    pref_df["Preference_Norm"] = pref_df["Preference"] / (pref_df["Preference"].abs().sum() + 1e-9)
    exposure_df = exposure_counts.rename("Exposure").reset_index().rename(columns={"index": "Category"})

    merged = pd.merge(pref_df, exposure_df, on="Category", how="outer").fillna(0)

    fig2, ax2 = plt.subplots()
    merged_plot = merged.sort_values("Exposure")
    ax2.plot(merged_plot["Category"], merged_plot["Exposure"], label="Exposure", marker="o")
    ax2.plot(merged_plot["Category"], merged_plot["Preference_Norm"], label="Normalized Preference", marker="o")
    ax2.set_xticklabels(merged_plot["Category"], rotation=45, ha="right")
    ax2.set_ylabel("Normalized Value")
    ax2.set_title("Exposure vs. User Preference")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("""
    This chart compares the **relative exposure** to content categories with the **user's expressed preferences**.
    Strong alignment suggests the system is personalized, while divergence may indicate recommendation bias.
    """)



    corr = merged["Exposure"].corr(merged["Preference_Norm"])
    if corr > 0.5:
        st.success(f"✅ Strong alignment between preferences and exposure (correlation = {round(corr, 2)}) — the system seems **personalized**.")
    elif corr > 0.2:
        st.info(f"ℹ️ Moderate alignment between preferences and exposure (correlation = {round(corr, 2)}).")
    else:
        st.warning(f"⚠️ Weak alignment (correlation = {round(corr, 2)}) — user preferences may not be reflected well in exposure.")

    # -------------------------------
    # 3. Gini Coefficient for Exposure Fairness
    # -------------------------------
    st.subheader("3. Gini Coefficient: Exposure Equality")

    def gini(array):
        array = array.flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)
        array = array.astype(float) + 1e-10
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    import numpy as np
    gini_value = gini(exposure_counts.values)

    st.metric("📊 Gini Coefficient", round(gini_value, 2))

    if gini_value > 0.5:
        st.warning("⚠️ High Gini suggests strong exposure inequality — certain categories dominate.")
    elif gini_value > 0.3:
        st.info("ℹ️ Moderate Gini — some imbalance in exposure, but not severe.")
    else:
        st.success("✅ Low Gini — exposure is fairly distributed across categories.")

    # -------------------------------
    # 4. Popularity Bias in Recommendations
    # -------------------------------
    st.subheader("4. Popularity Bias (Watch Count Distribution)")

    if st.session_state.watch_counts:
        watch_counts_df = pd.DataFrame(st.session_state.watch_counts.items(), columns=["Title", "Watch Count"])
        watch_counts_df = watch_counts_df.sort_values(by="Watch Count", ascending=False).reset_index(drop=True)

        fig4, ax4 = plt.subplots()
        sns.lineplot(x=watch_counts_df.index, y="Watch Count", data=watch_counts_df, color="red", ax=ax4)
        ax4.set_title("Content Popularity Distribution")
        ax4.set_xlabel("Content Index (Sorted)")
        ax4.set_ylabel("Watch Count")
        st.pyplot(fig4)

        top_10_prop = watch_counts_df.head(10)["Watch Count"].sum() / watch_counts_df["Watch Count"].sum()
        if top_10_prop > 0.5:
            st.warning(f"⚠️ Top 10 items make up {round(top_10_prop*100)}% of all views — strong **popularity bias** detected.")
        else:
            st.success("✅ Popularity is not overly skewed — system does not overly favor a few items.")
    else:
        st.info("No watch data to assess popularity bias.")
# -------------------------------
# 🧭 AUTONOMY TAB
# -------------------------------
with autonomy_tab:
    st.header("🧭 Autonomy Analysis")
    st.markdown("""
    This section explores **user autonomy** — how much control the user has in shaping recommendations
    based on explicit choices like likes/dislikes.
    """)

    interaction_df = pd.DataFrame([
        i if len(i) == 3 else i[:3] + [i[3]] for i in live_interactions
    ], columns=["Title", "Category", "Action", "Extra"] if any(len(i) > 3 for i in live_interactions) else ["Title", "Category", "Action"])
    autonomy_counts = interaction_df["Action"].value_counts()

    st.subheader("Interaction Types Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x=autonomy_counts.index, y=autonomy_counts.values, palette="Set2", ax=ax)
    ax.set_title("User Action Types")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    manual_interactions = autonomy_counts.get("liked", 0) + autonomy_counts.get("disliked", 0)
    skipped = autonomy_counts.get("skipped", 0)

    st.metric("Manual Interactions", manual_interactions)
    if manual_interactions > skipped:
        st.success("✅ The user is actively engaging by liking/disliking content, indicating strong autonomy.")
    elif skipped > manual_interactions:
        st.warning("⚠️ Skipped interactions dominate — user might not feel in control or lacks interest.")
    else:
        st.info("ℹ️ User engagement is balanced between active and passive actions.")
    st.metric("Skipped/Passive", skipped)

# -------------------------------
# 🔍 TRANSPARENCY TAB
# -------------------------------
with transparency_tab:
    st.header("🔍 Transparency Analysis")
    st.markdown("""
    This section evaluates **transparency** — whether users interact meaningfully and consistently
    with specific categories, indicating clarity in system behavior.
    """)

    if not interaction_df.empty:
        st.subheader("Category-wise Feedback")
        category_action_matrix = pd.crosstab(interaction_df["Category"], interaction_df["Action"])
        st.dataframe(category_action_matrix)

        st.subheader("Heatmap of Actions by Category")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(category_action_matrix, annot=True, cmap="YlGnBu", fmt="d", ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("No interaction data available.")

# -------------------------------
# 🌈 DIVERSITY TAB
# -------------------------------
with diversity_tab:
    st.header("🌈 Diversity Analysis")
    st.markdown("""
    This section explores the **diversity** of content exposure — are users exposed to a rich variety of categories?
    """)

    category_counts = interaction_df["Category"].value_counts()
    st.subheader("Interacted Categories (Live User)")
    fig3, ax3 = plt.subplots()
    sns.barplot(x=category_counts.index, y=category_counts.values, palette="pastel", ax=ax3)
    ax3.set_title("Diversity of Content Exposure")
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    def gini(array):
        array = array.flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)
        array = array.astype(np.float64) + 1e-10
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    diversity_score = 1 - gini(category_counts.values)
    st.metric("Diversity Score (1 - Gini)", round(diversity_score, 2))

    # Explanation for developers
    st.markdown("""
    The **diversity score** measures how evenly the user interacts with different content categories.
    A higher score (closer to 1) means balanced exposure across topics, indicating good variety.
    """)

    if diversity_score > 0.7:
        st.success("✅ High content diversity.")
    elif diversity_score > 0.4:
        st.info("ℹ️ Moderate diversity.")
    else:
        st.warning("⚠️ Low content diversity — user exposure may be narrow.")
