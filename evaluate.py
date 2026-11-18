"""Train/test evaluation for the Surprise collaborative filtering model."""
import pandas as pd
from surprise import Dataset, KNNBasic, Reader
from surprise.model_selection import train_test_split

from src.recommender import precision_recall_at_k


def evaluate(ratings: pd.DataFrame, test_size: float = 0.25, k: int = 5):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings[["user_id", "item_id", "rating"]].astype(
            {"user_id": str, "item_id": str}
        ), reader
    )
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
    algo = KNNBasic(
        k=20, min_k=1,
        sim_options={"name": "cosine", "user_based": True, "min_support": 1},
        verbose=False,
    )
    algo.fit(trainset)
    predictions = algo.test(testset)
    return precision_recall_at_k(predictions, k=k)


if __name__ == "__main__":
    data = pd.read_csv("data/ratings.csv")
    print("Evaluation:", evaluate(data, k=5))
