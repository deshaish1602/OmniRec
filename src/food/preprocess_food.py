import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

FOOD_CATEGORIES = {
    'desserts', 'breakfast', 'beverages', 'lunch',
    'main-dish', 'side-dishes', 'appetizers', 'snacks',
    'soups-stews', 'salads', 'breads', 'pasta-rice-and-grains',
    'meat', 'chicken', 'seafood', 'vegetables',
    'healthy', 'quick-breads', 'cookies-and-brownies',
    'cakes', 'pies-and-tarts', 'candy', 'frozen-desserts',
    'curries', 'stir-fry', 'sandwiches', 'pizza',
    'indian', 'italian', 'mexican', 'chinese', 'asian',
    'american', 'european', 'african', 'vegan', 'vegetarian'
}

def extract_best_tag(tags_str):
    try:
        tags_str = tags_str.strip("[]").replace("'", "")
        tags = [t.strip() for t in tags_str.split(',')]
        for tag in tags:
            if tag.lower() in FOOD_CATEGORIES:
                return tag.lower()
        for tag in tags:
            if tag not in ['60-minutes-or-less', '30-minutes-or-less',
                          '15-minutes-or-less', 'time-to-make',
                          'weeknight', 'general']:
                if len(tag) > 3:
                    return tag.lower()
        return tags[0].strip() if tags else "general"
    except:
        return "general"

def preprocess_food():
    print("Processing Food dataset...")
    df = pd.read_csv('data/food/RAW_recipes.csv', on_bad_lines='skip')
    df = df[['name', 'tags', 'description', 'ingredients']].copy()
    df.dropna(subset=['name'], inplace=True)
    df.fillna('', inplace=True)
    df['genres'] = df['tags'].apply(extract_best_tag)
    df['tags'] = (df['name']        + " " +
                  df['tags']        + " " +
                  df['description'] + " " +
                  df['ingredients'])
    df['tags'] = df['tags'].apply(clean_text)
    df.rename(columns={'name': 'title'}, inplace=True)
    df = df[['title', 'tags', 'genres']].head(20000).reset_index(drop=True)
    df.to_csv('data/food/processed_food.csv', index=False)
    print(f"Done! {len(df)} recipes saved!")
    print(f"\nTop genres: {df['genres'].value_counts().head(10).to_dict()}")
    return df

if __name__ == "__main__":
    preprocess_food()
