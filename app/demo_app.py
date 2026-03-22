import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import streamlit as st
import pandas as pd
import nltk
import re
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="OmniRec", page_icon="🎯", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; color: #e8e4d8; }
    .hero-title { font-size: 3.5rem; font-weight: 800; text-align: center; color: #ffffff; margin-bottom: 0.2rem; }
    .hero-subtitle { font-size: 1.1rem; text-align: center; color: #888; margin-bottom: 2rem; }
    .domain-card { background: #13111d; border: 1px solid #2a2535; border-radius: 16px; padding: 2rem; text-align: center; margin: 0.5rem; }
    .domain-name { font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-top: 0.5rem; }
    .domain-desc { font-size: 0.85rem; color: #888; margin-top: 0.3rem; }
    .rec-card { background: #13111d; border: 1px solid #2a2535; border-radius: 12px; padding: 1rem 1.5rem; margin: 0.5rem 0; }
    .rec-title { font-size: 1rem; font-weight: 600; color: #ffffff; }
    .rec-genre { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }
    .rec-score { font-size: 0.85rem; color: #e84040; font-weight: 600; }
    .score-bar-bg { background: #2a2535; border-radius: 10px; height: 6px; margin-top: 0.4rem; }
    .score-bar-fill { background: #e84040; border-radius: 10px; height: 6px; }
    .badge { display: inline-block; background: #1d0e0e; color: #e84040; border: 1px solid #e84040; border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.75rem; font-weight: 600; margin-right: 0.3rem; }
    .not-found-box { background: #1a1000; border: 1px solid #e89540; border-radius: 12px; padding: 1.2rem; color: #e89540; margin: 1rem 0; }
    div[data-testid="stButton"] > button { background: #e84040; color: white; border: none; border-radius: 8px; padding: 0.5rem 2rem; font-weight: 600; font-size: 1rem; width: 100%; }
</style>
""", unsafe_allow_html=True)
SAMPLE_DATA = {
    'movies': [
        {'title': 'Inception', 'tags': 'dream heist thriller sci-fi mind bending action adventure', 'genres': 'Sci-Fi & Fantasy'},
        {'title': 'The Dark Knight', 'tags': 'batman joker superhero action crime thriller gotham', 'genres': 'Action & Adventure'},
        {'title': 'Interstellar', 'tags': 'space time travel science black hole nasa astronaut', 'genres': 'Sci-Fi & Fantasy'},
        {'title': 'Parasite', 'tags': 'korean class society thriller drama family crime', 'genres': 'International Movies'},
        {'title': 'Dangal', 'tags': 'wrestling sports india father daughters biopic drama', 'genres': 'International Movies'},
        {'title': 'The Matrix', 'tags': 'virtual reality simulation action sci-fi hacker neo', 'genres': 'Sci-Fi & Fantasy'},
        {'title': 'Avengers Endgame', 'tags': 'superhero marvel action adventure time travel thanos', 'genres': 'Action & Adventure'},
        {'title': 'Joker', 'tags': 'dc comics villain origin drama psychological thriller', 'genres': 'Thrillers'},
        {'title': 'Money Heist', 'tags': 'heist robbery spain crime thriller drama action', 'genres': 'International Movies'},
        {'title': 'Squid Game', 'tags': 'korean survival game thriller drama competition money', 'genres': 'International Movies'},
        {'title': 'Titanic', 'tags': 'romance love ship disaster drama historical tragedy', 'genres': 'Romantic Movies'},
        {'title': 'The Notebook', 'tags': 'romance love story drama emotional couple relationship', 'genres': 'Romantic Movies'},
        {'title': 'Get Out', 'tags': 'horror thriller race mystery psychological suspense', 'genres': 'Horror Movies'},
        {'title': 'A Quiet Place', 'tags': 'horror survival monsters silence family thriller', 'genres': 'Horror Movies'},
        {'title': 'The Office', 'tags': 'comedy workplace office funny mockumentary humor', 'genres': 'TV Comedies'},
        {'title': 'Breaking Bad', 'tags': 'crime drama chemistry teacher drug meth thriller', 'genres': 'TV Dramas'},
        {'title': 'Stranger Things', 'tags': 'sci-fi horror supernatural kids mystery 80s thriller', 'genres': 'TV Sci-Fi & Fantasy'},
        {'title': 'Sacred Games', 'tags': 'india crime thriller drama gangster police mumbai', 'genres': 'International Movies'},
        {'title': 'Dark', 'tags': 'german time travel mystery thriller sci-fi family', 'genres': 'International Movies'},
        {'title': 'Narcos', 'tags': 'drug cartel colombia crime drama thriller pablo escobar', 'genres': 'TV Dramas'},
    ],
    'food': [
        {'title': 'Spaghetti Carbonara', 'tags': 'pasta italian egg cheese bacon cream dinner', 'genres': 'main-dish'},
        {'title': 'Chicken Biryani', 'tags': 'rice chicken indian spicy aromatic basmati curry', 'genres': 'main-dish'},
        {'title': 'Chocolate Lava Cake', 'tags': 'chocolate dessert cake warm gooey sweet baking', 'genres': 'desserts'},
        {'title': 'Caesar Salad', 'tags': 'salad lettuce chicken caesar dressing croutons healthy', 'genres': 'salads'},
        {'title': 'Margherita Pizza', 'tags': 'pizza italian tomato cheese basil dough bake', 'genres': 'main-dish'},
        {'title': 'Pancakes', 'tags': 'breakfast fluffy syrup butter morning sweet eggs', 'genres': 'breakfast'},
        {'title': 'Mango Smoothie', 'tags': 'smoothie mango fruit drink beverage healthy fresh', 'genres': 'beverages'},
        {'title': 'Butter Chicken', 'tags': 'chicken indian curry butter tomato cream spicy', 'genres': 'main-dish'},
        {'title': 'Tiramisu', 'tags': 'italian dessert coffee mascarpone cream sweet cake', 'genres': 'desserts'},
        {'title': 'Tacos', 'tags': 'mexican beef tortilla cheese salsa guacamole spicy', 'genres': 'main-dish'},
        {'title': 'Greek Salad', 'tags': 'salad feta olive cucumber tomato healthy mediterranean', 'genres': 'salads'},
        {'title': 'French Toast', 'tags': 'breakfast bread egg milk sweet syrup morning', 'genres': 'breakfast'},
        {'title': 'Pad Thai', 'tags': 'thai noodles shrimp peanut asian stir fry', 'genres': 'main-dish'},
        {'title': 'Cheesecake', 'tags': 'dessert cream cheese sweet baking cake strawberry', 'genres': 'desserts'},
        {'title': 'Green Tea', 'tags': 'tea drink beverage healthy antioxidant hot cold', 'genres': 'beverages'},
        {'title': 'Masala Dosa', 'tags': 'indian breakfast crispy rice potato spicy south', 'genres': 'breakfast'},
        {'title': 'Tom Yum Soup', 'tags': 'thai soup spicy sour shrimp mushroom lemon', 'genres': 'soups-stews'},
        {'title': 'Garlic Bread', 'tags': 'bread garlic butter italian side dish baked', 'genres': 'side-dishes'},
        {'title': 'Miso Soup', 'tags': 'japanese soup miso tofu seaweed light healthy', 'genres': 'soups-stews'},
        {'title': 'Brownie', 'tags': 'chocolate dessert sweet baking fudge nuts cake', 'genres': 'desserts'},
    ],
    'fashion': [
        {'title': 'White Cotton T-Shirt', 'tags': 'white cotton tshirt casual men apparel basic', 'genres': 'Apparel'},
        {'title': 'Blue Denim Jeans', 'tags': 'blue denim jeans casual men women bottomwear', 'genres': 'Apparel'},
        {'title': 'Black Leather Jacket', 'tags': 'black leather jacket men casual stylish outerwear', 'genres': 'Apparel'},
        {'title': 'Red Floral Dress', 'tags': 'red floral dress women casual summer ethnic', 'genres': 'Apparel'},
        {'title': 'White Sneakers', 'tags': 'white sneakers casual shoes men women sports', 'genres': 'Footwear'},
        {'title': 'Brown Leather Boots', 'tags': 'brown leather boots men casual formal footwear', 'genres': 'Footwear'},
        {'title': 'Black Heels', 'tags': 'black heels women formal party shoes footwear', 'genres': 'Footwear'},
        {'title': 'Sports Running Shoes', 'tags': 'sports running shoes men women athletic footwear', 'genres': 'Footwear'},
        {'title': 'Leather Handbag', 'tags': 'leather handbag women brown black accessories stylish', 'genres': 'Accessories'},
        {'title': 'Aviator Sunglasses', 'tags': 'aviator sunglasses men women gold black accessories', 'genres': 'Accessories'},
        {'title': 'Silver Watch', 'tags': 'silver watch men formal casual accessories metal', 'genres': 'Accessories'},
        {'title': 'Silk Saree', 'tags': 'silk saree women ethnic indian traditional apparel', 'genres': 'Apparel'},
        {'title': 'Formal White Shirt', 'tags': 'white formal shirt men office business apparel', 'genres': 'Apparel'},
        {'title': 'Navy Blazer', 'tags': 'navy blazer men formal office smart apparel', 'genres': 'Apparel'},
        {'title': 'Pink Kurta', 'tags': 'pink kurta women ethnic indian traditional apparel', 'genres': 'Apparel'},
        {'title': 'Canvas Backpack', 'tags': 'canvas backpack men women casual travel accessories', 'genres': 'Accessories'},
        {'title': 'Gold Earrings', 'tags': 'gold earrings women jewellery accessories ethnic', 'genres': 'Accessories'},
        {'title': 'Track Pants', 'tags': 'track pants men women sports casual bottomwear', 'genres': 'Apparel'},
        {'title': 'Perfume Woody', 'tags': 'perfume woody fragrance men personal care grooming', 'genres': 'Personal Care'},
        {'title': 'Face Moisturizer', 'tags': 'face moisturizer cream skincare personal care women', 'genres': 'Personal Care'},
    ]
}
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
@st.cache_resource
def build_model(domain):
    df = pd.DataFrame(SAMPLE_DATA[domain])
    df['clean_tags'] = df['tags'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=500)
    matrix = tfidf.fit_transform(df['clean_tags'])
    similarity = cosine_similarity(matrix)
    return df, similarity
def get_recommendations(query, domain, top_n=5):
    df, similarity = build_model(domain)
    query_clean = query.strip().lower()
    matches = df[df['title'].str.lower().str.contains(query_clean, na=False)]
    if matches.empty:
        matches = df[df['tags'].str.lower().str.contains(query_clean, na=False)]
    if matches.empty:
        return None, df.sample(min(top_n, len(df)))
    idx = matches.index[0]
    matched_title = df.loc[idx, 'title']
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    results = []
    for i, score in scores:
        results.append({'title': df.loc[i, 'title'], 'genre': df.loc[i, 'genres'], 'similarity': round(float(score) * 100, 1)})
    return matched_title, results
if 'domain' not in st.session_state:
    st.session_state.domain = None
def show_landing():
    st.markdown('<div class="hero-title">OmniRec</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">AI-Powered Multi-Domain Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Choose Your Domain")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="domain-card"><div style="font-size:3rem">🎬</div><div class="domain-name">Movies</div><div class="domain-desc">Discover films you will love</div></div>""", unsafe_allow_html=True)
        if st.button("Explore Movies", key="movies"):
            st.session_state.domain = "movies"
            st.rerun()
    with col2:
        st.markdown("""<div class="domain-card"><div style="font-size:3rem">🍽️</div><div class="domain-name">Food</div><div class="domain-desc">Find recipes you will enjoy</div></div>""", unsafe_allow_html=True)
        if st.button("Explore Food", key="food"):
            st.session_state.domain = "food"
            st.rerun()
    with col3:
        st.markdown("""<div class="domain-card"><div style="font-size:3rem">👗</div><div class="domain-name">Fashion</div><div class="domain-desc">Discover your style</div></div>""", unsafe_allow_html=True)
        if st.button("Explore Fashion", key="fashion"):
            st.session_state.domain = "fashion"
            st.rerun()
    st.markdown("---")
    st.info("Demo version — Full version runs locally with 8000+ movies, 20000+ recipes and 10000+ fashion items!")
def show_recommender(domain):
    placeholders = {'movies': 'e.g. Inception, Dangal, Parasite, Dark', 'food': 'e.g. pasta, biryani, chocolate cake', 'fashion': 'e.g. shirt, sneakers, handbag'}
    emojis = {'movies': '🎬', 'food': '🍽️', 'fashion': '👗'}
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("Back"):
            st.session_state.domain = None
            st.rerun()
    with col_title:
        st.markdown(f"## {emojis[domain]} {domain.title()} Recommendations")
    st.markdown("---")
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        user_input = st.text_input("Search", placeholder=placeholders[domain], label_visibility="collapsed")
    with col_btn:
        search_clicked = st.button("Search")
    with st.expander("Filter by Genre/Category"):
        df, _ = build_model(domain)
        genres = sorted(df['genres'].unique().tolist())
        selected_genre = st.selectbox("Select Genre", ["All"] + genres)
        if selected_genre != "All":
            filtered = df[df['genres'] == selected_genre]
            st.markdown(f"**Top items in '{selected_genre}':**")
            for _, row in filtered.head(5).iterrows():
                st.markdown(f"""<div class="rec-card"><div class="rec-title">{row['title']}</div><div class="rec-genre">{row['genres']}</div></div>""", unsafe_allow_html=True)
    if search_clicked and user_input:
        with st.spinner(f"Finding recommendations..."):
            matched, results = get_recommendations(user_input, domain)
        if matched is None:
            st.markdown(f"""<div class="not-found-box">"{user_input}" not found — showing popular suggestions!</div>""", unsafe_allow_html=True)
            for _, row in results.iterrows():
                st.markdown(f"""<div class="rec-card"><div class="rec-title">{row['title']}</div></div>""", unsafe_allow_html=True)
        else:
            st.success(f"Showing recommendations for: **{matched}**")
            st.markdown("### Top 5 Recommendations")
            for i, r in enumerate(results, 1):
                score = r['similarity']
                st.markdown(f"""
                <div class="rec-card">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <span class="badge">#{i}</span>
                            <span class="rec-title">{r['title']}</span>
                            <div class="rec-genre">{r['genre']}</div>
                        </div>
                        <div class="rec-score">{score}% match</div>
                    </div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{min(score,100)}%"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
    elif search_clicked and not user_input:
        st.warning("Please enter something to search!")
if st.session_state.domain is None:
    show_landing()
else:
    show_recommender(st.session_state.domain)
