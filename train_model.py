import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (you can change this to any dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
data = pd.read_csv(url, names=columns)

# Prepare data
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Save the trained model
joblib.dump(model, 'model.joblib')

# Function to take user input and predict flower species
def predict_flower():
    print("\nEnter flower measurements to predict its species:")
    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))

        # Reshape the input data into a 2D array for prediction
        user_input = [[sepal_length, sepal_width, petal_length, petal_width]]

        # Load the saved model and predict
        model = joblib.load('model.joblib')
        prediction = model.predict(user_input)

        print(f"Predicted flower species: {prediction[0]}")

    except ValueError:
        print("Invalid input. Please enter numeric values for the flower measurements.")

# Call the prediction function
predict_flower()
