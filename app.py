from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("house_price_model.pkl", "rb"))

# Feature names
FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert input to NumPy array
        input_features = [data[feature] for feature in FEATURES]
        input_array = np.array([input_features])

        # Predict
        predicted_price = model.predict(input_array)[0] * 100000

        # Convert NumPy float32 to Python float to fix JSON serialization issue
        predicted_price = float(predicted_price)

        # Get feature importance
        feature_importance = dict(zip(
            FEATURES,
            [float(val) for val in model.feature_importances_.round(3)]
        ))

        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'feature_importance': feature_importance
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
