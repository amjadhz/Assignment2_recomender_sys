import pandas as pd

# Load the dataset
df = pd.read_csv("data/bbc_recommender_dataset_clean.csv")

# Define the program lists
black_british_programs = [
    "Black to Life: Explore the lives of forgotten Black Britons",
    "Black and British: A Forgotten History",
    "Black Is the New Black",
    "A.Dot’s Story of Grime",
    "Black Nurses: The Women Who Saved the NHS",
    "Desert Island Discs",
    "Start the Week",
    "Saturday Live",
    "Ten Significant Events in Black British History",
    "Coping with burnout when you're Black",
    "Black Mental Health During Uncertain Times",
    "The Open University on Mental Health",
    "The Race Chat",
    "Race, Covid and Me",
    "Learning About Black Scottish History",
    "1Xtra Talks: Being Black in Britain",
    "1Xtra Talks: Identity and Culture",
    "The match that pitted white players against black players",
    "Radio 4’s The Listening Project",
    "Let’s Settle This",
    "Young, Gifted and Classical",
    "Will Britain Ever Have a Black PM?",
    "Roots, Reggae, Rebellion",
    "Ten black British artists to celebrate",
    "Back in Time for Brixton",
    "Free Thinking: Black British History on Radio 3",
    "Britain’s Black Past",
    "Caribbean Food Made Easy",
    "The Caribbean with Andi and Miquita",
    "1Xtra - Made In Britain: UKG",
    "The Unwanted: The Secret Windrush Files",
    "Coming to England",
    "Salt, by Selina Thompson",
    '"Leigh-Anne: Race, Pop & Power - Signed"',
    "A Musical Family Christmas with the Kanneh-Masons"
]

asian_british_programs = [
    "Man Like Mobeen",
    "Supercar, Superfam",
    "Are British Asians more socially conservative than rest of UK?",
    "My Asian Alter Ego",
    "British Asians in art",
    "Anita goes to Bollywood",
    "Britain’s first Asian plus-size model",
    "Being British Asian: Who Do We Think We Are?",
    "Searching for Mum",
    "The Big British Asian Summer on Three",
    "What goes on behind the doors of artist Raqib Shaw’s studio?",
    "When Bhangra is life",
    "Recipes that Made Me",
    "In Tune: The Bhavan perform a Raag Rageshree",
    "The Boy with the Topknot",
    "Why we need to talk about suicide in the Asian community",
    "BBC Radio 4’s The Listening Project",
    "A Passage to Britain",
    "A poppadom robot",
    "British Asians and identity",
    "Looking beyond the stereotypes of Asian men",
    "When will a British South Asian footballer play for England’s senior squad?",
    "My Asian Family – the Musical",
    "British Asian Experience",
    "Our NHS: A Hidden History",
    "The Colony",
    "Sir Mortimer and Magnus",
    "Loop",
    "Blinded by the Light",
    '"My Family,"',
    "Storyville - Afghan Cricket Club - Out of the Ashes",
    "India with Sanjeev Bhaskar - The Longest Road",
    "Madhur Jaffrey's Flavours of India - Punjab",
    "Subnormal: A British Scandal",
    "Small Axe",
    "Go Jetters",
    "Beena and Amrit",
    "Dance Passion - 2022: Leeds",
    
]

# Filter and tag
matched_black = df[df['title'].isin(black_british_programs)].copy()
matched_black['theme'] = 'Black British'

matched_asian = df[df['title'].isin(asian_british_programs)].copy()
matched_asian['theme'] = 'Asian British'

# Combine and save
matched = pd.concat([matched_black, matched_asian])
matched.to_csv("data/matched_british_programmes.csv", index=False)

print(f"Done! {len(matched)} programmes saved to data/matched_british_programmes.csv")
