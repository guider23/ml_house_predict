from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("house_price_model.pkl", "rb"))

# Input features expected in form
FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_features = [data[feature] for feature in FEATURES]
        input_array = np.array([input_features])
        predicted_price = model.predict(input_array)[0] * 100000

        feature_importance = dict(zip(FEATURES, model.feature_importances_.round(3).tolist()))

        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'feature_importance': feature_importance
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
