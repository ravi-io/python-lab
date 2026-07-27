"""
Program 11: Text Classification using Scikit-learn
Time Complexity: O(n * m) for TF-IDF + O(n * m * k) for Naive Bayes
Space Complexity: O(n * m) for document-term matrix
Requires: pip install scikit-learn
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# Driver Code
if __name__ == "__main__":
    # Sample text dataset
    texts = [
        "I love this movie, it is fantastic",
        "This film is wonderful and amazing",
        "Great movie with excellent acting",
        "Best film I have ever watched",
        "The movie was lovely and heartwarming",
        "This movie is terrible and boring",
        "Worst film ever, waste of time",
        "I hate this movie, very bad",
        "Awful acting and horrible plot",
        "Disappointing movie, not worth watching",
        "Amazing storyline and great direction",
        "Beautiful cinematography and superb cast",
        "Dull movie with no excitement",
        "Pathetic screenplay and poor direction",
        "Incredible performance by the lead actor",
        "Unbearable to watch, very dull",
    ]

    labels = [
        "positive", "positive", "positive", "positive",
        "positive", "negative", "negative", "negative",
        "negative", "negative", "positive", "positive",
        "negative", "negative", "positive", "negative",
    ]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42
    )

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Multinomial Naive Bayes
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = classifier.predict(X_test_tfidf)

    print("=== Text Classification using Scikit-learn ===\n")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Classify new text
    print("--- New Predictions ---")
    new_texts = [
        "This movie is absolutely wonderful",
        "Terrible film, very disappointing",
        "An average movie with some good moments",
    ]
    new_tfidf = vectorizer.transform(new_texts)
    predictions = classifier.predict(new_tfidf)

    for text, pred in zip(new_texts, predictions):
        print(f"  \"{text}\" -> {pred}")
