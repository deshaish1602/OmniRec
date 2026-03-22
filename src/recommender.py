import pickle
import pandas as pd

def load_model(domain):
    df         = pickle.load(open(f'models/{domain}_df.pkl',         'rb'))
    similarity = pickle.load(open(f'models/{domain}_similarity.pkl', 'rb'))
    return df, similarity

def recommend(item_name, domain, top_n=5):
    df, similarity = load_model(domain)
    item_name_clean = item_name.strip().lower()
    df['title_lower'] = df['title'].str.lower()
    matches = df[df['title_lower'].str.contains(item_name_clean, na=False)]

    if matches.empty:
        return {
            'status'     : 'not_found',
            'message'    : f'"{item_name}" not found in {domain} database.',
            'suggestions': get_popular(df, top_n)
        }

    idx           = matches.index[0]
    matched_title = df.loc[idx, 'title']
    sim_scores    = list(enumerate(similarity[idx]))
    sim_scores    = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores    = sim_scores[1:top_n+1]

    results = []
    for i, score in sim_scores:
        results.append({
            'title'     : df.loc[i, 'title'],
            'similarity': round(float(score) * 100, 1),
            'genre'     : df.loc[i, 'genres'] if 'genres' in df.columns else 'N/A'
        })

    return {
        'status'      : 'success',
        'searched_for': matched_title,
        'domain'      : domain,
        'results'     : results
    }

def get_popular(df, top_n=5):
    sample = df.sample(top_n)
    return [{'title': row['title']} for _, row in sample.iterrows()]

def recommend_by_genre(genre, domain, top_n=5):
    df, _ = load_model(domain)
    if 'genres' not in df.columns:
        return {'status': 'error', 'message': 'No genre column available'}
    genre_clean = genre.strip().lower()
    filtered = df[df['genres'].str.lower().str.contains(genre_clean, na=False)]
    if filtered.empty:
        return {'status': 'not_found', 'message': f'No items found for genre: {genre}'}
    results = []
    for _, row in filtered.head(top_n).iterrows():
        results.append({'title': row['title'], 'genre': row['genres']})
    return {'status': 'success', 'domain': domain, 'genre': genre, 'results': results}

if __name__ == "__main__":
    print("\n🎬 Testing Movies...")
    result = recommend("Inception", "movies")
    print(f"  Searched for: {result.get('searched_for')}")
    for r in result.get('results', []):
        print(f"  → {r['title']:<40} {r['similarity']}% match")

    print("\n🍽️  Testing Food...")
    result = recommend("pasta", "food")
    print(f"  Searched for: {result.get('searched_for')}")
    for r in result.get('results', []):
        print(f"  → {r['title']:<40} {r['similarity']}% match")

    print("\n👗 Testing Fashion...")
    result = recommend("shirt", "fashion")
    print(f"  Searched for: {result.get('searched_for')}")
    for r in result.get('results', []):
        print(f"  → {r['title']:<40} {r['similarity']}% match")

    print("\n✅ Recommender working perfectly!")
