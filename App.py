from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the pickle file
model_path = os.path.join(current_dir, 'cancer.pkl')

# Load the model with error handling
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Model file 'cancer.pkl' not found. Please ensure it's in the correct directory.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("Index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.form['features']
        features = features.split(',')
        np_features = np.asarray(features, dtype=np.float32)

        # Prediction
        pred = model.predict(np_features.reshape(1, -1))
        message = ['Cancrouse' if pred[0] == 1 else 'Not Cancrouse']
    except Exception as e:
        message = [f"Error in prediction: {e}"]

    return render_template('Index.html', message=message)

# Python main
if __name__ == '__main__':
    app.run(debug=True)
