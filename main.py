"""Run the sample collaborative filtering recommender."""
import pandas as pd

from src.recommender import CollaborativeFilteringRecommender
from src.evaluate import evaluate


def main():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")

    recommender = CollaborativeFilteringRecommender(ratings, movies).fit()

    user_id = "1"
    print(f"\nTop recommendations for existing user {user_id}:")
    for row in recommender.recommend(user_id, top_n=5):
        print(f"  {row['title']:<28} predicted rating: {row['estimated_rating']}")

    print(f"\nUsers most similar to user {user_id}:")
    for row in recommender.user_similarity(user_id, top_n=5):
        print(f"  User {row['user_id']}: similarity {row['similarity']}")

    new_user_ratings = {"1": 5, "3": 4, "5": 2}
    print("\nRecommendations for a new user who rated items 1, 3 and 5:")
    for row in recommender.recommend_for_new_user(new_user_ratings, top_n=5):
        print(f"  {row['title']:<28} predicted rating: {row['estimated_rating']}")

    print("\nOffline evaluation:")
    print(evaluate(ratings, k=5))


if __name__ == "__main__":
    main()
