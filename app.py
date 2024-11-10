# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return "Welcome to the ML model API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data from POST request
    features = np.array(data['features']).reshape(1, -1)  # Reshape input to match model input

    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
