"""
Program 19: Visualize Decision Boundaries
Time Complexity: O(n * k * grid_size^2) for KNN on grid
Space Complexity: O(grid_size^2) for mesh grid
Requires: pip install matplotlib scikit-learn numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def plot_decision_boundary(X, y, classifier, title, ax):
    """Plot decision boundary for a classifier."""
    h = 0.02  # Step size in mesh

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict on mesh grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
               edgecolors='black', s=30)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

# Driver Code
if __name__ == "__main__":
    print("=== Decision Boundary Visualization ===\n")

    # Generate datasets
    np.random.seed(42)
    X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1,
                                         factor=0.5, random_state=42)

    # Classifiers
    classifiers = [
        ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
        ("SVM (RBF)", SVC(kernel='rbf', gamma='auto')),
        ("Decision Tree", DecisionTreeClassifier(max_depth=4)),
        ("Naive Bayes", GaussianNB()),
    ]

    # Create plots for moons dataset
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Decision Boundaries - Moons Dataset", fontsize=14)

    for ax, (name, clf) in zip(axes.ravel(), classifiers):
        clf.fit(X_moons, y_moons)
        score = clf.score(X_moons, y_moons)
        plot_decision_boundary(X_moons, y_moons, clf,
                               f"{name} (acc={score:.2f})", ax)

    plt.tight_layout()
    plt.savefig('decision_boundaries_moons.png', dpi=100, bbox_inches='tight')
    print("Saved: decision_boundaries_moons.png")

    # Create plots for circles dataset
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Decision Boundaries - Circles Dataset", fontsize=14)

    for ax, (name, clf) in zip(axes.ravel(), classifiers):
        clf.fit(X_circles, y_circles)
        score = clf.score(X_circles, y_circles)
        plot_decision_boundary(X_circles, y_circles, clf,
                               f"{name} (acc={score:.2f})", ax)

    plt.tight_layout()
    plt.savefig('decision_boundaries_circles.png', dpi=100, bbox_inches='tight')
    print("Saved: decision_boundaries_circles.png")

    # Print accuracy comparison
    print("\n--- Accuracy Comparison ---")
    print(f"{'Classifier':<20} {'Moons':<10} {'Circles':<10}")
    print("-" * 40)
    for name, clf in classifiers:
        clf.fit(X_moons, y_moons)
        moon_acc = clf.score(X_moons, y_moons)
        clf.fit(X_circles, y_circles)
        circ_acc = clf.score(X_circles, y_circles)
        print(f"{name:<20} {moon_acc:<10.3f} {circ_acc:<10.3f}")

    plt.show()
