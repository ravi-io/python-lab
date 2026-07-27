"""
Program 8: Naive Bayes Classifier (Gaussian Naive Bayes)
Time Complexity: O(n * m) for training, O(m * k) for prediction
  where n = samples, m = features, k = classes
Space Complexity: O(m * k)
"""

import math
from collections import defaultdict


class NaiveBayes:
    """Gaussian Naive Bayes Classifier."""

    def __init__(self):
        self.classes = []
        self.mean = defaultdict(list)
        self.var = defaultdict(list)
        self.priors = {}

    def fit(self, X, y):
        """Train the classifier."""
        self.classes = list(set(y))
        n_samples = len(y)

        for cls in self.classes:
            # Get samples belonging to this class
            X_cls = [X[i] for i in range(n_samples) if y[i] == cls]
            n_cls = len(X_cls)
            self.priors[cls] = n_cls / n_samples

            # Calculate mean and variance for each feature
            n_features = len(X[0])
            for j in range(n_features):
                values = [X_cls[i][j] for i in range(n_cls)]
                mean = sum(values) / n_cls
                var = sum((x - mean) ** 2 for x in values) / n_cls + 1e-9
                self.mean[cls].append(mean)
                self.var[cls].append(var)

    def _gaussian_pdf(self, x, mean, var):
        """Calculate Gaussian probability density."""
        return (1 / math.sqrt(2 * math.pi * var)) * math.exp(-(x - mean) ** 2 / (2 * var))

    def predict_single(self, sample):
        """Predict class for a single sample."""
        posteriors = {}
        for cls in self.classes:
            posterior = math.log(self.priors[cls])
            for i, x in enumerate(sample):
                pdf = self._gaussian_pdf(x, self.mean[cls][i], self.var[cls][i])
                posterior += math.log(pdf + 1e-300)
            posteriors[cls] = posterior
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        """Predict classes for multiple samples."""
        return [self.predict_single(sample) for sample in X]

    def accuracy(self, X, y):
        """Calculate classification accuracy."""
        predictions = self.predict(X)
        correct = sum(1 for p, actual in zip(predictions, y) if p == actual)
        return correct / len(y)


# Driver Code
if __name__ == "__main__":
    # Iris-like dataset (sepal_length, sepal_width, petal_length, petal_width)
    X_train = [
        [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
        [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
        [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5],
        [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1],
        [6.5, 3.0, 5.8, 2.2], [7.6, 3.0, 6.6, 2.1],
    ]
    y_train = ['Setosa'] * 5 + ['Versicolor'] * 5 + ['Virginica'] * 5

    X_test = [
        [5.0, 3.4, 1.5, 0.2],
        [6.7, 3.1, 4.4, 1.4],
        [6.9, 3.1, 5.4, 2.1],
    ]
    y_test = ['Setosa', 'Versicolor', 'Virginica']

    print("=== Naive Bayes Classifier ===\n")
    clf = NaiveBayes()
    clf.fit(X_train, y_train)

    print("Training Accuracy:", f"{clf.accuracy(X_train, y_train) * 100:.1f}%")
    print("\nPredictions on test data:")
    predictions = clf.predict(X_test)
    for sample, pred, actual in zip(X_test, predictions, y_test):
        status = "✓" if pred == actual else "✗"
        print(f"  {sample} -> Predicted: {pred:12s} Actual: {actual:12s} {status}")
