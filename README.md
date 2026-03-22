# OmniRec - Multi-Domain Recommendation System

An AI-powered recommendation engine built with NLP that works across 3 domains: Movies, Food, and Fashion.

## Features
- Multi-domain recommendations (Movies, Food, Fashion)
- TF-IDF vectorization + Cosine Similarity
- Genre/Category filtering
- Unknown input handling with fallback suggestions
- Clean Streamlit UI

## Tech Stack
- Python 3.13
- scikit-learn (TF-IDF, Cosine Similarity)
- NLTK (Text preprocessing)
- Streamlit (UI)
- Pandas & NumPy

## Project Structure
```
OmniRec/
├── data/
│   ├── movies/
│   ├── food/
│   └── fashion/
├── src/
│   ├── movies/preprocess_movies.py
│   ├── food/preprocess_food.py
│   ├── fashion/preprocess_fashion.py
│   ├── features.py
│   └── recommender.py
├── app/
│   └── streamlit_app.py
├── models/
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/OmniRec.git
cd OmniRec
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
- Movies: Netflix Titles Dataset (Kaggle)
- Food: Food.com Recipes Dataset (Kaggle)
- Fashion: Fashion Products Dataset (Kaggle)

Place them in their respective data/ folders.

### 5. Preprocess data
```bash
python3 src/movies/preprocess_movies.py
python3 src/food/preprocess_food.py
python3 src/fashion/preprocess_fashion.py
```

### 6. Build models
```bash
python3 -c "from src.features import build_features; build_features('movies'); build_features('food'); build_features('fashion')"
```

### 7. Run the app
```bash
streamlit run app/streamlit_app.py
```

## How It Works
1. Text features are extracted from each domain dataset
2. TF-IDF converts text into numerical vectors
3. Cosine Similarity finds the most similar items
4. Top 5 recommendations are returned with similarity scores

## Interview Topics
- Content-Based Filtering
- TF-IDF Vectorization
- Cosine Similarity
- NLP Text Preprocessing
- Streamlit Deployment

## Author
Aishwarya Deshwal
