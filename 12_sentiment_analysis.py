"""
Program 12: Sentiment Analysis using Python
Time Complexity: O(n * m) where n = documents, m = avg words per doc
Space Complexity: O(V) where V = vocabulary size
Requires: pip install scikit-learn
"""

import re
from collections import defaultdict
import math


class SentimentAnalyzer:
    """Sentiment Analyzer using bag-of-words + logistic-like scoring."""

    def __init__(self):
        self.word_scores = {}
        self.positive_words = set()
        self.negative_words = set()

    def _tokenize(self, text):
        """Simple tokenizer: lowercase and split on non-alpha."""
        return re.findall(r'[a-z]+', text.lower())

    def fit(self, texts, labels):
        """Train sentiment model using word frequency ratios."""
        word_pos_count = defaultdict(int)
        word_neg_count = defaultdict(int)
        total_pos = 0
        total_neg = 0

        for text, label in zip(texts, labels):
            tokens = self._tokenize(text)
            if label == "positive":
                total_pos += 1
                for word in set(tokens):
                    word_pos_count[word] += 1
            else:
                total_neg += 1
                for word in set(tokens):
                    word_neg_count[word] += 1

        # Calculate log-likelihood ratio for each word
        vocab = set(word_pos_count.keys()) | set(word_neg_count.keys())
        for word in vocab:
            pos_rate = (word_pos_count[word] + 1) / (total_pos + 2)
            neg_rate = (word_neg_count[word] + 1) / (total_neg + 2)
            self.word_scores[word] = math.log(pos_rate / neg_rate)

        # Identify strongly positive/negative words
        for word, score in self.word_scores.items():
            if score > 0.3:
                self.positive_words.add(word)
            elif score < -0.3:
                self.negative_words.add(word)

    def predict_single(self, text):
        """Predict sentiment for a single text."""
        tokens = self._tokenize(text)
        score = sum(self.word_scores.get(word, 0) for word in tokens)
        if score > 0:
            return "positive", score
        else:
            return "negative", score

    def predict(self, texts):
        """Predict sentiment for multiple texts."""
        return [self.predict_single(text)[0] for text in texts]

    def accuracy(self, texts, labels):
        """Calculate accuracy."""
        predictions = self.predict(texts)
        correct = sum(1 for p, a in zip(predictions, labels) if p == a)
        return correct / len(labels)


# Driver Code
if __name__ == "__main__":
    # Training data
    train_texts = [
        "I love this product, it works great",
        "Amazing quality and fast delivery",
        "Best purchase I ever made, highly recommend",
        "Wonderful experience, very satisfied",
        "Excellent service and great value",
        "This is the worst product ever",
        "Terrible quality, broke after one day",
        "Very disappointed, waste of money",
        "Horrible customer service, never again",
        "Bad experience, do not buy this",
        "Fantastic features and easy to use",
        "Superb design and perfect functionality",
        "Awful product, completely useless",
        "Pathetic quality, total garbage",
    ]
    train_labels = [
        "positive", "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative", "negative",
        "positive", "positive", "negative", "negative",
    ]

    # Test data
    test_texts = [
        "I really love this amazing product",
        "This is terrible and disappointing",
        "Great quality and wonderful design",
        "Worst purchase, horrible experience",
        "Good value for money, satisfied",
    ]
    test_labels = ["positive", "negative", "positive", "negative", "positive"]

    print("=== Sentiment Analysis ===\n")

    analyzer = SentimentAnalyzer()
    analyzer.fit(train_texts, train_labels)

    print(f"Training Accuracy: {analyzer.accuracy(train_texts, train_labels)*100:.1f}%")
    print(f"Test Accuracy: {analyzer.accuracy(test_texts, test_labels)*100:.1f}%")

    print("\n--- Predictions ---")
    for text in test_texts:
        sentiment, score = analyzer.predict_single(text)
        print(f"  \"{text}\"")
        print(f"    -> {sentiment} (score: {score:.3f})\n")

    print("--- Key Sentiment Words ---")
    top_pos = sorted(analyzer.positive_words,
                     key=lambda w: analyzer.word_scores[w], reverse=True)[:5]
    top_neg = sorted(analyzer.negative_words,
                     key=lambda w: analyzer.word_scores[w])[:5]
    print(f"  Positive: {', '.join(top_pos)}")
    print(f"  Negative: {', '.join(top_neg)}")
