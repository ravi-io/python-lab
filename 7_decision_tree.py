"""
Program 7: Decision Tree Learning (ID3 Algorithm)
Time Complexity: O(n * m * d) where n = samples, m = features, d = tree depth
Space Complexity: O(n * m)
"""

import math
from collections import Counter


def entropy(labels):
    """Calculate entropy of a label set."""
    total = len(labels)
    if total == 0:
        return 0
    counts = Counter(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def information_gain(data, labels, feature_idx):
    """Calculate information gain for a feature."""
    parent_entropy = entropy(labels)

    # Get unique values for this feature
    values = set(row[feature_idx] for row in data)
    weighted_entropy = 0

    for value in values:
        subset_labels = [labels[i] for i in range(len(data)) if data[i][feature_idx] == value]
        weighted_entropy += (len(subset_labels) / len(labels)) * entropy(subset_labels)

    return parent_entropy - weighted_entropy


def build_tree(data, labels, features, feature_names):
    """Build decision tree using ID3 algorithm."""
    # Base cases
    if len(set(labels)) == 1:
        return labels[0]

    if not features:
        return Counter(labels).most_common(1)[0][0]

    # Select best feature based on information gain
    gains = [(f, information_gain(data, labels, f)) for f in features]
    best_feature = max(gains, key=lambda x: x[1])[0]

    tree = {feature_names[best_feature]: {}}
    values = set(row[best_feature] for row in data)

    remaining_features = [f for f in features if f != best_feature]

    for value in values:
        subset_data = [data[i] for i in range(len(data)) if data[i][best_feature] == value]
        subset_labels = [labels[i] for i in range(len(data)) if data[i][best_feature] == value]

        if not subset_data:
            tree[feature_names[best_feature]][value] = Counter(labels).most_common(1)[0][0]
        else:
            tree[feature_names[best_feature]][value] = build_tree(
                subset_data, subset_labels, remaining_features, feature_names
            )

    return tree


def classify(tree, sample, feature_names):
    """Classify a sample using the decision tree."""
    if not isinstance(tree, dict):
        return tree

    feature = list(tree.keys())[0]
    feature_idx = feature_names.index(feature)
    value = sample[feature_idx]

    if value in tree[feature]:
        return classify(tree[feature][value], sample, feature_names)
    return "Unknown"


def print_tree(tree, indent=""):
    """Display the decision tree."""
    if not isinstance(tree, dict):
        print(f"{indent}-> {tree}")
        return
    for feature, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}[{feature} = {value}]")
            print_tree(subtree, indent + "  ")


# Driver Code
if __name__ == "__main__":
    # PlayTennis Dataset
    feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']

    data = [
        ['Sunny', 'Hot', 'High', 'Weak'],
        ['Sunny', 'Hot', 'High', 'Strong'],
        ['Overcast', 'Hot', 'High', 'Weak'],
        ['Rain', 'Mild', 'High', 'Weak'],
        ['Rain', 'Cool', 'Normal', 'Weak'],
        ['Rain', 'Cool', 'Normal', 'Strong'],
        ['Overcast', 'Cool', 'Normal', 'Strong'],
        ['Sunny', 'Mild', 'High', 'Weak'],
        ['Sunny', 'Cool', 'Normal', 'Weak'],
        ['Rain', 'Mild', 'Normal', 'Weak'],
        ['Sunny', 'Mild', 'Normal', 'Strong'],
        ['Overcast', 'Mild', 'High', 'Strong'],
        ['Overcast', 'Hot', 'Normal', 'Weak'],
        ['Rain', 'Mild', 'High', 'Strong'],
    ]

    labels = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
              'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

    # Build tree
    features = list(range(len(feature_names)))
    tree = build_tree(data, labels, features, feature_names)

    print("=== Decision Tree (ID3) ===")
    print("\nDecision Tree Structure:")
    print_tree(tree)

    # Test predictions
    print("\n--- Predictions ---")
    test_samples = [
        ['Sunny', 'Cool', 'High', 'Strong'],
        ['Overcast', 'Mild', 'Normal', 'Weak'],
        ['Rain', 'Hot', 'Normal', 'Weak'],
    ]

    for sample in test_samples:
        prediction = classify(tree, sample, feature_names)
        print(f"  {dict(zip(feature_names, sample))} -> Play: {prediction}")
