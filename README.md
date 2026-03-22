# OmniRec — AI-Powered Multi-Domain Recommendation System

OmniRec is an end-to-end intelligent recommendation system that leverages Natural Language Processing (NLP) to generate personalized suggestions across multiple domains including Movies, Food, and Fashion.

The system is designed to simulate a real-world recommendation engine where unstructured text data is processed, transformed into feature vectors, and used to generate similarity-based recommendations through a unified pipeline.

---

## Live Demo
 https://omnirec-5yk945k5xe7upciphycbjz.streamlit.app/

---

## Key Features

- Multi-domain recommendation system (Movies, Food, Fashion)
- Content-based filtering using NLP techniques
- TF-IDF vectorization for feature extraction
- Cosine similarity for recommendation ranking
- Robust handling of unknown or noisy inputs
- Category and genre-based filtering
- Interactive web interface using Streamlit
- Modular and scalable pipeline architecture

---

## Project Architecture

The system follows a modular machine learning pipeline:

User Input → Text Processing → Feature Engineering → Similarity Computation → Ranked Recommendations → Output UI

### Components

**Backend Recommendation Engine**
Handles feature extraction, similarity computation, and recommendation logic.

**Text Processing Module**
Performs tokenization, stopword removal, and normalization using NLTK.

**Feature Engineering Module**
Transforms textual data into numerical vectors using TF-IDF.

**Similarity Engine**
Uses cosine similarity to compute relationships between items.

**Streamlit Frontend**
Provides an interactive interface for users to input queries and view recommendations.

---

## Tech Stack

- Python
- scikit-learn
- NLTK
- Pandas
- NumPy
- Streamlit

---

## Project Structure
```
OmniRec/
│
├── data/
│   ├── movies/
│   ├── food/
│   └── fashion/
│
├── src/
│   ├── movies/
│   ├── food/
│   ├── fashion/
│   ├── features.py
│   └── recommender.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository
```
git clone https://github.com/deshaish1602/OmniRec.git
cd OmniRec
```

### 2. Create virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Prepare datasets

Download datasets from Kaggle:
- Movies → Netflix dataset
- Food → Recipes dataset
- Fashion → Product dataset

Place them in:
- data/movies/
- data/food/
- data/fashion/

### 5. Preprocess data
```
python3 src/movies/preprocess_movies.py
python3 src/food/preprocess_food.py
python3 src/fashion/preprocess_fashion.py
```

### 6. Build feature models
```
python3 -c "from src.features import build_features; build_features('movies'); build_features('food'); build_features('fashion')"
```

### 7. Run the application
```
streamlit run app/streamlit_app.py
```

---

## Example Workflow

1. User enters a query (e.g. "romantic movies" or "summer outfits")
2. System preprocesses the input text
3. TF-IDF transforms input into vector representation
4. Cosine similarity finds similar items from dataset
5. Top-N recommendations are generated
6. Results are displayed in the Streamlit interface

---

## Use Cases

- Personalized recommendation systems
- Content-based filtering applications
- NLP-based search engines
- Cross-domain recommendation engines
- Educational projects in machine learning and AI

---

## Future Improvements

- Hybrid recommendation system (content + collaborative filtering)
- User preference learning and personalization
- Integration with large language models (LLMs)
- Cloud deployment (AWS / GCP)
- Real-time recommendation APIs
- Vector database integration (FAISS / Pinecone)

---

## Author

Aishwarya Deshwal
B.Tech Computer Science Engineering
Bennett University
