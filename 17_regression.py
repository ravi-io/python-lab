"""
Program 17: Regression Analysis (Linear & Polynomial Regression)
Time Complexity: O(n * m^2) for linear regression (normal equation)
Space Complexity: O(n * m)
"""

import math


class LinearRegression:
    """Simple Linear Regression using least squares."""

    def __init__(self):
        self.slope = 0
        self.intercept = 0

    def fit(self, X, y):
        """Fit using ordinary least squares."""
        n = len(X)
        sum_x = sum(X)
        sum_y = sum(y)
        sum_xy = sum(x * yi for x, yi in zip(X, y))
        sum_x2 = sum(x ** 2 for x in X)

        # Slope and intercept formulas
        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        self.intercept = (sum_y - self.slope * sum_x) / n

    def predict(self, X):
        """Predict values."""
        return [self.slope * x + self.intercept for x in X]

    def r_squared(self, X, y):
        """Calculate R-squared (coefficient of determination)."""
        y_pred = self.predict(X)
        mean_y = sum(y) / len(y)
        ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def mse(self, X, y):
        """Mean Squared Error."""
        y_pred = self.predict(X)
        return sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred)) / len(y)

class PolynomialRegression:
    """Polynomial Regression using gradient descent."""

    def __init__(self, degree=2, lr=0.01, epochs=1000):
        self.degree = degree
        self.lr = lr
        self.epochs = epochs
        self.coefficients = None

    def _transform(self, X):
        """Transform X into polynomial features."""
        return [[x ** d for d in range(self.degree + 1)] for x in X]

    def fit(self, X, y):
        """Fit using gradient descent."""
        X_poly = self._transform(X)
        n = len(X)
        self.coefficients = [0.0] * (self.degree + 1)

        for _ in range(self.epochs):
            # Compute predictions
            predictions = [sum(c * xp for c, xp in zip(self.coefficients, xi))
                           for xi in X_poly]

            # Compute gradients
            for j in range(self.degree + 1):
                gradient = sum((predictions[i] - y[i]) * X_poly[i][j]
                               for i in range(n)) / n
                self.coefficients[j] -= self.lr * gradient

    def predict(self, X):
        """Predict values."""
        X_poly = self._transform(X)
        return [sum(c * xp for c, xp in zip(self.coefficients, xi))
                for xi in X_poly]

    def r_squared(self, X, y):
        """Calculate R-squared."""
        y_pred = self.predict(X)
        mean_y = sum(y) / len(y)
        ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


# Driver Code
if __name__ == "__main__":
    print("=== Regression Analysis ===\n")

    # Linear Regression
    print("--- Simple Linear Regression ---")
    # Study hours vs marks
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [45, 50, 55, 60, 62, 67, 70, 75, 80, 85]

    lr = LinearRegression()
    lr.fit(X, y)

    print(f"  Equation: y = {lr.slope:.2f}x + {lr.intercept:.2f}")
    print(f"  R-squared: {lr.r_squared(X, y):.4f}")
    print(f"  MSE: {lr.mse(X, y):.4f}")
    print(f"\n  Predictions:")
    test_hours = [3, 6, 11]
    for h in test_hours:
        pred = lr.predict([h])[0]
        print(f"    Hours={h} -> Predicted Marks={pred:.1f}")

    # Polynomial Regression
    print("\n--- Polynomial Regression (degree=2) ---")
    # Non-linear data
    X2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y2 = [2.5, 5.1, 10.2, 17.0, 25.5, 36.1, 49.0, 64.2, 81.0, 100.5]

    poly = PolynomialRegression(degree=2, lr=0.0001, epochs=5000)
    poly.fit(X2, y2)

    print(f"  Coefficients: {[f'{c:.3f}' for c in poly.coefficients]}")
    print(f"  (c0 + c1*x + c2*x^2)")
    print(f"  R-squared: {poly.r_squared(X2, y2):.4f}")
    print(f"\n  Predictions:")
    for x in [3, 7, 12]:
        pred = poly.predict([x])[0]
        print(f"    x={x} -> y={pred:.2f}")
