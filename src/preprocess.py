import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import ast
import re
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def extract_names(json_str, key='name', limit=3):
    try:
        items = ast.literal_eval(json_str)
        return " ".join([i[key] for i in items[:limit]])
    except:
        return ""

# ─────────────────────────────────────────
# MOVIES PREPROCESSING
# ─────────────────────────────────────────

def preprocess_movies():
    print("\n🎬 Preprocessing Movies...")
    df = pd.read_csv("data/movies/tmdb_5000_movies.csv")
    df = df[['title', 'overview', 'genres', 'keywords']].copy()
    df.dropna(subset=['title', 'overview'], inplace=True)
    df['genres']   = df['genres'].apply(lambda x: extract_names(x))
    df['keywords'] = df['keywords'].apply(lambda x: extract_names(x))
    df['tags'] = (df['overview'] + " " +
                  df['genres']   + " " +
                  df['keywords'])
    df['tags'] = df['tags'].apply(clean_text)
    df = df[['title', 'tags', 'genres']].reset_index(drop=True)
    df.to_csv("data/movies/processed_movies.csv", index=False)
    print(f"  Done! {len(df)} movies saved. ✅")
    return df

# ─────────────────────────────────────────
# FOOD PREPROCESSING
# ─────────────────────────────────────────

def preprocess_food():
    print("\n🍽️  Preprocessing Food...")
    df = pd.read_csv("data/food/RAW_recipes.csv", on_bad_lines='skip')
    df = df[['name', 'tags', 'description', 'ingredients']].copy()
    df.dropna(subset=['name'], inplace=True)
    df.fillna('', inplace=True)
    df['tags'] = (df['name']        + " " +
                  df['tags']        + " " +
                  df['description'] + " " +
                  df['ingredients'])
    df['tags'] = df['tags'].apply(clean_text)
    df = df[['name', 'tags']].copy()
    df.rename(columns={'name': 'title'}, inplace=True)
    df = df.head(20000).reset_index(drop=True)
    df.to_csv("data/food/processed_food.csv", index=False)
    print(f"  Done! {len(df)} recipes saved. ✅")
    return df

# ─────────────────────────────────────────
# FASHION PREPROCESSING
# ─────────────────────────────────────────

def preprocess_fashion():
    print("\n👗 Preprocessing Fashion...")
    df = pd.read_csv("data/fashion/styles.csv", on_bad_lines='skip')
    cols = ['productDisplayName', 'masterCategory',
            'subCategory', 'articleType',
            'baseColour', 'usage']
    df = df[cols].copy()
    df.dropna(subset=['productDisplayName'], inplace=True)
    df.fillna('', inplace=True)
    df['tags'] = (df['productDisplayName'] + " " +
                  df['masterCategory']     + " " +
                  df['subCategory']        + " " +
                  df['articleType']        + " " +
                  df['baseColour']         + " " +
                  df['usage'])
    df['tags'] = df['tags'].apply(clean_text)
    df.rename(columns={'productDisplayName': 'title',
                       'masterCategory': 'genres'}, inplace=True)
    df = df[['title', 'tags', 'genres']].reset_index(drop=True)
    df.to_csv("data/fashion/processed_fashion.csv", index=False)
    print(f"  Done! {len(df)} products saved. ✅")
    return df

# ─────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────

if __name__ == "__main__":
    preprocess_movies()
    preprocess_food()
    preprocess_fashion()
    print("\n✅ All datasets preprocessed successfully!")
