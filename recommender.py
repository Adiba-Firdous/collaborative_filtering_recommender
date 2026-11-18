"""Collaborative filtering recommender built with the Surprise library."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd
from surprise import Dataset, KNNBasic, Reader


class CollaborativeFilteringRecommender:
    """User-based CF for known users and item-based fold-in for new users."""

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None):
        required = {"user_id", "item_id", "rating"}
        missing = required - set(ratings.columns)
        if missing:
            raise ValueError(f"ratings is missing required columns: {sorted(missing)}")

        self.ratings = ratings.copy()
        self.ratings["user_id"] = self.ratings["user_id"].astype(str)
        self.ratings["item_id"] = self.ratings["item_id"].astype(str)
        self.ratings["rating"] = pd.to_numeric(self.ratings["rating"], errors="raise")
        if self.ratings.empty:
            raise ValueError("ratings cannot be empty")
        if not self.ratings["rating"].between(1, 5).all():
            raise ValueError("ratings must be between 1 and 5")

        self.movies = movies.copy() if movies is not None else pd.DataFrame(
            columns=["item_id", "title"]
        )
        if not self.movies.empty:
            self.movies["item_id"] = self.movies["item_id"].astype(str)
        self.user_algo = None
        self.item_algo = None
        self.trainset = None
        self.item_trainset = None

    def fit(self) -> "CollaborativeFilteringRecommender":
        """Train user-user and item-item cosine similarity models."""
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            self.ratings[["user_id", "item_id", "rating"]], reader
        )
        self.trainset = dataset.build_full_trainset()

        user_sim = {"name": "cosine", "user_based": True, "min_support": 1}
        item_sim = {"name": "cosine", "user_based": False, "min_support": 1}

        self.user_algo = KNNBasic(k=20, min_k=1, sim_options=user_sim, verbose=False)
        self.user_algo.fit(self.trainset)

        # A second Surprise model provides item-item similarities for new-user fold-in.
        self.item_algo = KNNBasic(k=20, min_k=1, sim_options=item_sim, verbose=False)
        self.item_algo.fit(self.trainset)
        self.item_trainset = self.item_algo.trainset
        return self

    def _check_fitted(self):
        if self.user_algo is None or self.item_algo is None:
            raise RuntimeError("Call fit() before requesting recommendations.")

    def _title(self, item_id: str) -> str:
        if not self.movies.empty:
            match = self.movies.loc[self.movies["item_id"] == str(item_id), "title"]
            if not match.empty:
                return str(match.iloc[0])
        return str(item_id)

    def user_similarity(self, user_id: str, top_n: int = 5) -> list[dict]:
        """Return the most similar known users and their similarity scores."""
        self._check_fitted()
        raw = str(user_id)
        try:
            inner = self.trainset.to_inner_uid(raw)
        except ValueError:
            return []
        similarities = self.user_algo.sim[inner]
        neighbors = [
            (self.trainset.to_raw_uid(i), float(score))
            for i, score in enumerate(similarities) if i != inner
        ]
        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        return [{"user_id": uid, "similarity": round(score, 4)}
                for uid, score in neighbors[:top_n]]

    def recommend(self, user_id: str, top_n: int = 5) -> list[dict]:
        """Recommend unrated items for a user already present in training data."""
        self._check_fitted()
        uid = str(user_id)
        known_users = set(self.ratings["user_id"])
        if uid not in known_users:
            raise ValueError(
                f"Unknown user {uid!r}. Use recommend_for_new_user() instead."
            )

        already_rated = set(
            self.ratings.loc[self.ratings["user_id"] == uid, "item_id"]
        )
        all_items = sorted(set(self.ratings["item_id"]))
        candidates = [iid for iid in all_items if iid not in already_rated]

        predictions = [
            (iid, float(self.user_algo.predict(uid, iid).est)) for iid in candidates
        ]
        predictions.sort(key=lambda pair: pair[1], reverse=True)
        return [
            {"item_id": iid, "title": self._title(iid), "estimated_rating": round(score, 3)}
            for iid, score in predictions[:top_n]
        ]

    def recommend_for_new_user(
        self, user_ratings: dict[str, float], top_n: int = 5
    ) -> list[dict]:
        """
        Recommend for a user absent from training data.

        user_ratings maps item_id to the new user's 1-5 rating. The method uses
        Surprise's learned item-item cosine similarities to score unseen items.
        """
        self._check_fitted()
        if not user_ratings:
            raise ValueError("Provide at least one item rating for a new user.")

        normalized = {str(k): float(v) for k, v in user_ratings.items()}
        if any(v < 1 or v > 5 for v in normalized.values()):
            raise ValueError("All ratings must be between 1 and 5.")

        known_items = set(self.ratings["item_id"])
        unknown = set(normalized) - known_items
        if unknown:
            raise ValueError(f"Unknown item_id(s): {sorted(unknown)}")

        candidates = sorted(known_items - set(normalized))
        scored = []
        global_mean = float(self.ratings["rating"].mean())

        for candidate in candidates:
            try:
                candidate_inner = self.item_trainset.to_inner_iid(candidate)
            except ValueError:
                continue

            weighted_sum = 0.0
            similarity_sum = 0.0
            for rated_item, rating in normalized.items():
                rated_inner = self.item_trainset.to_inner_iid(rated_item)
                similarity = float(self.item_algo.sim[candidate_inner][rated_inner])
                if similarity > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += similarity

            score = weighted_sum / similarity_sum if similarity_sum else global_mean
            scored.append((candidate, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            {"item_id": iid, "title": self._title(iid), "estimated_rating": round(score, 3)}
            for iid, score in scored[:top_n]
        ]


def precision_recall_at_k(predictions: Iterable, k: int = 5, threshold: float = 4.0):
    """Calculate mean Precision@K and Recall@K for Surprise test predictions."""
    if k <= 0:
        raise ValueError("k must be positive")

    by_user = defaultdict(list)
    for prediction in predictions:
        uid, _, true_rating, estimated, _ = prediction
        by_user[uid].append((float(estimated), float(true_rating)))

    precisions, recalls = [], []
    for user_predictions in by_user.values():
        user_predictions.sort(key=lambda pair: pair[0], reverse=True)
        top_k = user_predictions[:k]
        relevant = sum(true >= threshold for _, true in user_predictions)
        hits = sum(true >= threshold for _, true in top_k)
        precisions.append(hits / k)
        if relevant:
            recalls.append(hits / relevant)

    return {
        "precision_at_k": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "k": k,
        "relevance_threshold": threshold,
    }
