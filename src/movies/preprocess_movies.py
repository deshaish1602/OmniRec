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

def preprocess_movies():
    print("Processing Netflix/Movies dataset...")
    df = pd.read_csv('data/movies/netflix_titles.csv')
    df = df.fillna('Unknown')
    print(f"Total titles : {len(df)}")
    df['tags'] = (df['title']       + " " +
                  df['description'] + " " +
                  df['listed_in']   + " " +
                  df['cast']        + " " +
                  df['director']    + " " +
                  df['country'])
    df['tags']   = df['tags'].apply(clean_text)
    df['genres'] = df['listed_in']
    df = df[['title', 'tags', 'genres']].reset_index(drop=True)
    df.to_csv('data/movies/processed_movies.csv', index=False)
    print(f"Done! {len(df)} titles saved!")
    return df

if __name__ == "__main__":
    preprocess_movies()
