from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import io
import os
import pickle
import joblib

app = Flask(__name__, static_url_path='/static')


# Load diabetes model (scikit-learn)
try:
    # Load the pickle model for text-based inputs
    diabetes_model = joblib.load('diabetes_model_logistic.pkl')
    print("Diabetes model loaded successfully")
except Exception as e:
    print(f"Error loading diabetes model: {e}")
    diabetes_model = None


def prepare_diabetes_features(form_data):
    """
    Process the form data into features expected by the scikit-learn diabetes model.
    Features required: Pregnancies, Age, BloodPressure, SkinThickness, Glucose,
    Insulin, BMI, DiabetesPedigreeFunction
    """
    try:
        # Extract the specific fields from form data and convert to appropriate types
        # This matches the exact features the diabetes model was trained on
        features = {
            'Pregnancies': float(form_data.get('pregnancies', 0)),
            'Age': float(form_data.get('age', 0)),
            'BloodPressure': float(form_data.get('bloodpressure', 0)),
            'SkinThickness': float(form_data.get('skinthickness', 0)),
            'Glucose': float(form_data.get('glucose', 0)),
            'Insulin': float(form_data.get('insulin', 0)),
            'BMI': float(form_data.get('bmi', 0)),
            'DiabetesPedigreeFunction': float(form_data.get('diabetespedigreefunction', 0))
        }

        # Convert to list in the right order for the model
        # This order must match the order of features the model was trained with
        feature_names = ['Pregnancies', 'Age', 'BloodPressure', 'SkinThickness',
                         'Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']

        # Create feature array in correct order
        feature_array = [features[name] for name in feature_names]

        print(f"Prepared diabetes features: {feature_array}")
        return np.array([feature_array])  # Return as 2D array for sklearn

    except Exception as e:
        print(f"Error preparing diabetes features: {e}")
        # Return None to indicate error
        return None


@app.route('/')
def index():
    # Serve the landing page
    return render_template('index.html')





@app.route('/diabetes')
def diabetes():
    return render_template('diabetes_prediction_page.html')


@app.route('/static/<path:path>')
def serve_static(path):
    # Serve static files (CSS, JS, images)
    return send_from_directory('static', path)


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if not diabetes_model:
        return jsonify({'error': 'Diabetes model not loaded'}), 500

    try:
        # Extract all form data for the diabetes prediction
        form_data = request.form

        # Prepare features for the scikit-learn model
        features = prepare_diabetes_features(form_data)

        if features is None:
            return jsonify({'error': 'Could not process the input data'}), 400

        # Make prediction with scikit-learn model
        if hasattr(diabetes_model, 'predict_proba'):
            # If model supports probabilities
            probabilities = diabetes_model.predict_proba(features)[0]
            prediction = diabetes_model.predict(features)[0]

            # Get confidence (probability of predicted class)
            confidence = probabilities[1] * 100 if prediction == 1 else probabilities[0] * 100

            result = {
                'label': 'Positive: Diabetes Likely' if prediction == 1 else 'Negative: Diabetes Unlikely',
                'confidence': round(confidence, 2)
            }
        else:
            # If model doesn't support probabilities
            prediction = diabetes_model.predict(features)[0]
            result = {
                'label': 'Positive: Diabetes Likely' if prediction == 1 else 'Negative: Diabetes Unlikely',
                'confidence': None  # No confidence score available
            }

        print(f"Diabetes prediction: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"Error during diabetes prediction: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)