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

def preprocess_fashion():
    print("Processing Fashion dataset...")
    df = pd.read_csv('data/fashion/styles.csv', on_bad_lines='skip')
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

    # Limit to 10,000 for speed
    df = df.head(10000)

    df.to_csv('data/fashion/processed_fashion.csv', index=False)
    print(f"Done! {len(df)} products saved!")
    print(f"Categories: {df['genres'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    preprocess_fashion()
