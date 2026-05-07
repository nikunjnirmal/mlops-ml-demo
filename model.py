# Import necessary libraries for machine learning and data handling
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset, which contains measurements of iris flowers
# X contains the features (sepal length, sepal width, petal length, petal width)
# y contains the target labels (species: setosa, versicolor, virginica)
X, y = load_iris(return_X_y=True)

# Split the dataset into training and testing sets
# 80% for training, 20% for testing, with a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier model
# This is an ensemble learning method that builds multiple decision trees
clf = RandomForestClassifier(n_estimators=150)

# Train the model using the training data
# The model learns to predict the iris species based on the flower measurements
clf.fit(X_train, y_train)

# Save the trained model to a file for later use
# This allows the model to be loaded and used for predictions without retraining
joblib.dump(clf, "iris_model.pkl")

# Print a confirmation message indicating the model has been saved
print("Model trained and saved as iris_model.pkl")
