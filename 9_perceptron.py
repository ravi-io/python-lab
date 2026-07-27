"""
Program 9: Perceptron Learning Algorithm
Time Complexity: O(epochs * n * m) where n = samples, m = features
Space Complexity: O(m) for weights
"""


class Perceptron:
    """Single-layer Perceptron classifier."""

    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def _step(self, x):
        """Step activation function."""
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        """Train the perceptron."""
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        errors_per_epoch = []

        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                # Compute output
                linear = sum(w * x for w, x in zip(self.weights, xi))
                prediction = self._step(linear + self.bias)

                # Update weights if prediction is wrong
                error = target - prediction
                if error != 0:
                    errors += 1
                    for j in range(n_features):
                        self.weights[j] += self.lr * error * xi[j]
                    self.bias += self.lr * error

            errors_per_epoch.append(errors)
            if errors == 0:
                print(f"  Converged at epoch {epoch + 1}")
                break

        return errors_per_epoch

    def predict(self, X):
        """Predict class labels."""
        predictions = []
        for xi in X:
            linear = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
            predictions.append(self._step(linear))
        return predictions

    def accuracy(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        return correct / len(y)


# Driver Code
if __name__ == "__main__":
    print("=== Perceptron Learning Algorithm ===\n")

    # AND Gate
    print("--- AND Gate ---")
    X_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_and = [0, 0, 0, 1]

    p_and = Perceptron(learning_rate=0.1, epochs=100)
    p_and.fit(X_and, y_and)
    print(f"  Weights: {p_and.weights}, Bias: {p_and.bias}")
    print(f"  Predictions: {p_and.predict(X_and)}")
    print(f"  Expected:    {y_and}\n")

    # OR Gate
    print("--- OR Gate ---")
    X_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_or = [0, 1, 1, 1]

    p_or = Perceptron(learning_rate=0.1, epochs=100)
    p_or.fit(X_or, y_or)
    print(f"  Weights: {p_or.weights}, Bias: {p_or.bias}")
    print(f"  Predictions: {p_or.predict(X_or)}")
    print(f"  Expected:    {y_or}\n")

    # NAND Gate
    print("--- NAND Gate ---")
    X_nand = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_nand = [1, 1, 1, 0]

    p_nand = Perceptron(learning_rate=0.1, epochs=100)
    p_nand.fit(X_nand, y_nand)
    print(f"  Weights: {p_nand.weights}, Bias: {p_nand.bias}")
    print(f"  Predictions: {p_nand.predict(X_nand)}")
    print(f"  Expected:    {y_nand}")
