
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
texts = [
    "I love this movie",
    "This is amazing",
    "I hate this movie",
    "This is terrible"
]

labels = ["Positive", "Positive", "Negative", "Negative"]

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# Test new text
test = ["This movie is amazing"]
test_data = vectorizer.transform(test)

# Predict sentiment
result = model.predict(test_data)

print("Sentiment:", result[0])
