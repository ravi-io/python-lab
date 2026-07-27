"""
Program 20: Recommendation System (Collaborative Filtering)
Time Complexity: O(n * m) for similarity, O(n * m * k) for predictions
  where n=users, m=items, k=similar users
Space Complexity: O(n * m) for rating matrix
"""

import math
from collections import defaultdict


class CollaborativeFilter:
    """User-based Collaborative Filtering Recommendation System."""

    def __init__(self, k_neighbors=3):
        self.k = k_neighbors
        self.ratings = {}
        self.user_means = {}
        self.similarity_cache = {}

    def fit(self, ratings_dict):
        """
        Load user ratings.
        ratings_dict: {user: {item: rating, ...}, ...}
        """
        self.ratings = ratings_dict
        self.user_means = {}
        for user, items in self.ratings.items():
            values = list(items.values())
            self.user_means[user] = sum(values) / len(values)

    def cosine_similarity(self, user1, user2):
        """Compute cosine similarity between two users."""
        key = (min(user1, user2), max(user1, user2))
        if key in self.similarity_cache:
            return self.similarity_cache[key]

        # Find common items
        common = set(self.ratings[user1]) & set(self.ratings[user2])
        if not common:
            return 0

        # Compute adjusted cosine similarity
        mean1 = self.user_means[user1]
        mean2 = self.user_means[user2]

        num = sum((self.ratings[user1][i] - mean1) *
                  (self.ratings[user2][i] - mean2) for i in common)
        den1 = math.sqrt(sum((self.ratings[user1][i] - mean1) ** 2
                             for i in common))
        den2 = math.sqrt(sum((self.ratings[user2][i] - mean2) ** 2
                             for i in common))

        sim = num / (den1 * den2) if den1 * den2 != 0 else 0
        self.similarity_cache[key] = sim
        return sim

    def predict_rating(self, user, item):
        """Predict rating for a user-item pair."""
        if user not in self.ratings:
            return self.user_means.get(user, 3.0)
        if item in self.ratings[user]:
            return self.ratings[user][item]

        # Find k most similar users who rated this item
        similarities = []
        for other_user in self.ratings:
            if other_user != user and item in self.ratings[other_user]:
                sim = self.cosine_similarity(user, other_user)
                if sim > 0:
                    similarities.append((other_user, sim))

        # Sort by similarity, take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:self.k]

        if not top_k:
            return self.user_means[user]

        # Weighted average
        num = sum(sim * (self.ratings[other][item] - self.user_means[other])
                  for other, sim in top_k)
        den = sum(abs(sim) for _, sim in top_k)

        return self.user_means[user] + (num / den if den != 0 else 0)

    def recommend(self, user, n=5):
        """Get top-n recommendations for a user."""
        if user not in self.ratings:
            return []

        # Find items not yet rated by user
        all_items = set()
        for items in self.ratings.values():
            all_items.update(items.keys())

        unrated = all_items - set(self.ratings[user].keys())

        # Predict ratings for unrated items
        predictions = []
        for item in unrated:
            pred = self.predict_rating(user, item)
            predictions.append((item, pred))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


# Driver Code
if __name__ == "__main__":
    print("=== Recommendation System (Collaborative Filtering) ===\n")

    # User-Item ratings (1-5 scale)
    ratings = {
        'Alice':   {'Movie1': 5, 'Movie2': 3, 'Movie3': 4, 'Movie5': 4},
        'Bob':     {'Movie1': 3, 'Movie2': 1, 'Movie3': 2, 'Movie4': 3, 'Movie5': 3},
        'Charlie': {'Movie1': 4, 'Movie2': 3, 'Movie3': 4, 'Movie4': 5, 'Movie5': 3},
        'David':   {'Movie1': 3, 'Movie2': 3, 'Movie3': 1, 'Movie4': 5, 'Movie5': 4},
        'Eve':     {'Movie1': 1, 'Movie2': 5, 'Movie3': 5, 'Movie4': 2, 'Movie5': 1},
    }

    # Display rating matrix
    all_items = sorted(set(item for user_ratings in ratings.values()
                           for item in user_ratings))
    print("Rating Matrix:")
    print(f"{'User':<10}", end="")
    for item in all_items:
        print(f"{item:<9}", end="")
    print()
    for user, user_ratings in ratings.items():
        print(f"{user:<10}", end="")
        for item in all_items:
            r = user_ratings.get(item, '-')
            print(f"{str(r):<9}", end="")
        print()

    # Build recommender
    recommender = CollaborativeFilter(k_neighbors=3)
    recommender.fit(ratings)

    # Show user similarities
    print("\n--- User Similarities ---")
    users = list(ratings.keys())
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            sim = recommender.cosine_similarity(users[i], users[j])
            if abs(sim) > 0.1:
                print(f"  {users[i]:8s} <-> {users[j]:8s}: {sim:.3f}")

    # Generate recommendations
    print("\n--- Recommendations ---")
    for user in ratings:
        recs = recommender.recommend(user, n=3)
        if recs:
            rec_str = ", ".join(f"{item}({score:.1f})" for item, score in recs)
            print(f"  {user}: {rec_str}")
        else:
            print(f"  {user}: Already rated all items")
