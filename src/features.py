import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# ─────────────────────────────────────────
# BUILD TF-IDF + SIMILARITY MATRIX
# ─────────────────────────────────────────

def build_features(domain):
    print(f"\n⚙️  Building features for: {domain}")

    # Load correct dataset
    paths = {
        'movies' : 'data/movies/processed_movies.csv',
        'food'   : 'data/food/processed_food.csv',
        'fashion': 'data/fashion/processed_fashion.csv'
    }

    df = pd.read_csv(paths[domain])
    df.dropna(subset=['tags'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # TF-IDF Vectorization
    print(f"  Running TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    print(f"  Matrix shape: {tfidf_matrix.shape}")

    # Cosine Similarity
    print(f"  Computing cosine similarity...")
    similarity = cosine_similarity(tfidf_matrix)
    print(f"  Similarity matrix shape: {similarity.shape}")

    # Save everything to models/
    os.makedirs('models', exist_ok=True)
    pickle.dump(df,         open(f'models/{domain}_df.pkl',          'wb'))
    pickle.dump(tfidf,      open(f'models/{domain}_tfidf.pkl',       'wb'))
    pickle.dump(similarity, open(f'models/{domain}_similarity.pkl',  'wb'))

    print(f"  Saved to models/ ✅")
    return df, tfidf, similarity

if __name__ == "__main__":
    build_features('movies')
    build_features('food')
    build_features('fashion')
    print("\n✅ All features built and saved!")
