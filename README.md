# Collaborative Filtering Movie Recommender (Surprise)

A small, runnable recommendation-system project using **Python** and the
**Surprise** library. It demonstrates user-item ratings, cosine similarity,
top-N recommendations, Precision@K / Recall@K evaluation, and recommendations
for a new user.

## Features

- Builds a user-item rating matrix from `data/ratings.csv`.
- Trains a user-based KNN collaborative filtering model with cosine similarity.
- Finds similar users.
- Generates top-N recommendations for an existing user (excluding items already rated).
- Uses item-item cosine similarity for a simple new-user fold-in recommendation.
- Evaluates a held-out test set with Precision@K and Recall@K.
- Includes a small sample movie dataset so the project can run immediately.

## Project structure

```text
collaborative_filtering_recommender/
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── src/
│   ├── __init__.py
│   ├── recommender.py
│   └── evaluate.py
├── tests/
│   └── test_recommender.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (Windows)

Open a terminal in this project folder.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> `scikit-surprise` includes compiled extensions. Python 3.10 or 3.11 is a
> practical choice. If installation fails on a newer Python version, use Python
> 3.11 in a fresh virtual environment.

## Run

```bash
python main.py
```

The program prints recommendations for an existing user, similar users,
recommendations for a new user, and evaluation metrics.

## Run tests

```bash
pip install pytest
pytest -q
```

## How it works

1. **User-item rating matrix:** rows are users, columns are items, and values are ratings.
2. **Similarity:** Surprise `KNNBasic` calculates cosine similarity between users.
3. **Top-N:** the model predicts scores for items the selected user has not rated and sorts them.
4. **Precision@K:** among the top K predicted items, the fraction that are relevant (rating >= 4).
5. **New users:** a new user supplies a few item ratings. Item-item cosine similarities learned by Surprise are used to estimate scores for unseen items.

## Customize

- Add rows to `data/ratings.csv` using columns `user_id,item_id,rating`.
- Add item names to `data/movies.csv` using columns `item_id,title`.
- Ratings must be from 1 to 5.
- In `main.py`, change `user_id` or `new_user_ratings` to try different examples.

## Important limitation

This is a learning/demo dataset, not a production recommender. New-user
recommendations require at least one rating for a known item. A real system
would need more data, stronger validation, and a strategy for completely
unrated users (for example, popularity-based recommendations).
