import pandas as pd
import pytest

from src.recommender import CollaborativeFilteringRecommender


@pytest.fixture
def model():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    return CollaborativeFilteringRecommender(ratings, movies).fit()


def test_existing_user_recommendations(model):
    results = model.recommend("1", top_n=3)
    assert len(results) <= 3
    assert all("estimated_rating" in row for row in results)


def test_new_user_recommendations(model):
    results = model.recommend_for_new_user({"1": 5, "3": 4}, top_n=3)
    assert len(results) <= 3
    assert all(row["item_id"] not in {"1", "3"} for row in results)


def test_unknown_existing_user_raises(model):
    with pytest.raises(ValueError):
        model.recommend("not-a-user")
