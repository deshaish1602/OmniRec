import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommender import recommend, recommend_by_genre, load_model

st.set_page_config(page_title="OmniRec", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; color: #e8e4d8; }
    .hero-title { font-size: 3.5rem; font-weight: 800; text-align: center; color: #ffffff; margin-bottom: 0.2rem; }
    .hero-subtitle { font-size: 1.1rem; text-align: center; color: #888; margin-bottom: 2rem; }
    .domain-card { background: #13111d; border: 1px solid #2a2535; border-radius: 16px; padding: 2rem; text-align: center; margin: 0.5rem; }
    .domain-emoji { font-size: 3rem; }
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
    div[data-testid="stButton"] > button:hover { background: #c73030; }
</style>
""", unsafe_allow_html=True)

if 'domain' not in st.session_state:
    st.session_state.domain = None

DOMAIN_GENRES = {
    'movies': [
        'Action & Adventure', 'Anime Features', 'Children & Family Movies',
        'Classic Movies', 'Comedies', 'Cult Movies', 'Documentaries',
        'Dramas', 'Faith & Spirituality', 'Horror Movies',
        'Independent Movies', 'International Movies', 'LGBTQ Movies',
        'Music & Musicals', 'Romantic Movies', 'Sci-Fi & Fantasy',
        'Sports Movies', 'Stand-Up Comedy', 'Thrillers',
        'TV Action & Adventure', 'TV Comedies', 'TV Dramas',
        'TV Horror', 'TV Mysteries', 'TV Sci-Fi & Fantasy',
        'TV Thrillers', 'Docuseries', 'Reality TV',
        'British TV Shows', 'International TV Shows',
        'Romantic TV Shows', 'Kids TV', 'Anime Series',
        'Crime TV Shows', 'Talk Shows & Variety'
    ],
    'food': [
        'beverages', 'breakfast', 'course', 'desserts',
        'lunch', 'main-dish', 'side-dishes', 'snacks',
        'appetizers', 'salads', 'soups-stews', 'pasta-rice-and-grains',
        'meat', 'chicken', 'seafood', 'vegetables',
        'healthy', 'vegan', 'vegetarian', 'curries',
        'italian', 'mexican', 'indian', 'chinese', 'asian'
    ],
    'fashion': [
        'Apparel', 'Accessories', 'Footwear',
        'Personal Care', 'Sporting Goods', 'Free Items', 'Home'
    ]
}

def get_genre_list(df, domain):
    defined = DOMAIN_GENRES.get(domain, [])
    actual = set()
    for genre_str in df['genres'].dropna():
        for g in str(genre_str).split(','):
            g = g.strip()
            if g and g != 'Unknown':
                actual.add(g)
    final = [g for g in defined if g in actual]
    if not final:
        final = sorted(list(actual))[:40]
    return final

def show_landing():
    st.markdown('<div class="hero-title">OmniRec</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Your AI-Powered Multi-Domain Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Choose Your Domain")
    st.markdown(" ")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="domain-card"><div class="domain-emoji">🎬</div><div class="domain-name">Movies</div><div class="domain-desc">Discover films you'll love based on your favourites</div></div>""", unsafe_allow_html=True)
        if st.button("Explore Movies", key="movies"):
            st.session_state.domain = "movies"
            st.rerun()
    with col2:
        st.markdown("""<div class="domain-card"><div class="domain-emoji">🍽️</div><div class="domain-name">Food</div><div class="domain-desc">Find recipes similar to dishes you already enjoy</div></div>""", unsafe_allow_html=True)
        if st.button("Explore Food", key="food"):
            st.session_state.domain = "food"
            st.rerun()
    with col3:
        st.markdown("""<div class="domain-card"><div class="domain-emoji">👗</div><div class="domain-name">Fashion</div><div class="domain-desc">Find clothing and accessories that match your style</div></div>""", unsafe_allow_html=True)
        if st.button("Explore Fashion", key="fashion"):
            st.session_state.domain = "fashion"
            st.rerun()

def show_recommender(domain):
    domain_info = {
        'movies': {'emoji': '🎬', 'label': 'Movies',  'placeholder': 'e.g. Inception, Dangal, Parasite, Money Heist'},
        'food':   {'emoji': '🍽️', 'label': 'Food',    'placeholder': 'e.g. pasta, chocolate cake, biryani'},
        'fashion':{'emoji': '👗', 'label': 'Fashion',  'placeholder': 'e.g. shirt, blue jeans, sneakers'},
    }
    info = domain_info[domain]
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("Back"):
            st.session_state.domain = None
            st.rerun()
    with col_title:
        st.markdown(f"## {info['emoji']} {info['label']} Recommendations")
    st.markdown("---")
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        user_input = st.text_input("Search", placeholder=info['placeholder'], label_visibility="collapsed")
    with col_btn:
        search_clicked = st.button("Search")
    with st.expander("Filter by Genre/Category"):
        df, _ = load_model(domain)
        if 'genres' in df.columns:
            genre_list = get_genre_list(df, domain)
            selected_genre = st.selectbox("Select Genre", ["All"] + genre_list)
            if selected_genre != "All":
                filtered = df[df['genres'].str.contains(selected_genre, na=False)]
                if not filtered.empty:
                    st.markdown(f"**Top items in '{selected_genre}':**")
                    for _, row in filtered.head(5).iterrows():
                        st.markdown(f"""<div class="rec-card"><div class="rec-title">{row['title']}</div><div class="rec-genre">{row['genres']}</div></div>""", unsafe_allow_html=True)
                else:
                    st.warning(f"No items found for: {selected_genre}")
    if search_clicked and user_input:
        with st.spinner(f"Finding recommendations for '{user_input}'..."):
            result = recommend(user_input, domain, top_n=5)
        if result['status'] == 'not_found':
            st.markdown(f"""<div class="not-found-box">Not found: {result['message']}<br><small>Showing popular suggestions instead:</small></div>""", unsafe_allow_html=True)
            for s in result.get('suggestions', []):
                st.markdown(f"""<div class="rec-card"><div class="rec-title">{s['title']}</div></div>""", unsafe_allow_html=True)
        else:
            st.success(f"Showing recommendations for: **{result['searched_for']}**")
            st.markdown("### Top 5 Recommendations")
            for i, r in enumerate(result['results'], 1):
                score = r['similarity']
                st.markdown(f"""
                <div class="rec-card">
                    <div style="display:flex; justify-content:space-between; align-items:center">
                        <div>
                            <span class="badge">#{i}</span>
                            <span class="rec-title">{r['title']}</span>
                            <div class="rec-genre">{r.get('genre', '')}</div>
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
