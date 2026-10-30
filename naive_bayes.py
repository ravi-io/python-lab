
from sklearn.naive_bayes import GaussianNB

# Training data
X = [[1, 2], [2, 3], [3, 3], [6, 7], [7, 8], [8, 8]]
y = ["Yes", "Yes", "Yes", "No", "No", "No"]

# Create and train the model
model = GaussianNB()
model.fit(X, y)

# Predict new data
result = model.predict([[2, 2]])

print("Predicted Class:", result[0])
