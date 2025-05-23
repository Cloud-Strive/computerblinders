from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
import joblib
import pandas as pd
import os
from django.conf import settings 
from django.views.decorators.cache import cache_page

# Create your views here.
def home_page(request):
    return render(request, 'home_page.html')


# load models
base_dir = settings.BASE_DIR # Project root
model_path = os.path.join(base_dir, 'prediction_app','saved_model', 'heart_disease_model.joblib')
scaler_path = os.path.join(base_dir, 'prediction_app','saved_model', 'heart_disease_scaler.joblib')
features_path = os.path.join(base_dir,'prediction_app', 'saved_model', 'heart_disease_features.joblib')


# catch any error before rendering app and models
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    print("Model, Scaler, and Features loaded successfully.")
    MODEL_LOADED = True
except Exception as e:
    print(f"unable to load model/scaler/features: {e}")
    model = None
    scaler = None
    feature_names = None
    MODEL_LOADED = False
    model_load_error = None  # To store any loading error


@csrf_protect 
def heart_page(request):
    context = {'prediction_text': "", 'prediction_proba_text': ""} 

    if request.method == 'POST':
        if not MODEL_LOADED:
            context['prediction_text'] = "Error: Model not loaded properly."
            return render(request, 'heart.html', context) 

        try:
            # 1. Get data from form
            input_data = {feature: request.POST.get(feature) for feature in feature_names}

            # 2. input field for heart disease causes
            input_data['age'] = int(input_data['age'])
            input_data['sex'] = int(input_data['sex'])
            input_data['cp'] = int(input_data['cp'])
            input_data['trestbps'] = int(input_data['trestbps'])
            input_data['chol'] = int(input_data['chol'])
            input_data['fbs'] = int(input_data['fbs'])
            input_data['restecg'] = int(input_data['restecg'])
            input_data['thalach'] = int(input_data['thalach'])
            input_data['exang'] = int(input_data['exang'])
            input_data['oldpeak'] = float(input_data['oldpeak'])
            input_data['slope'] = int(input_data['slope'])
            input_data['ca'] = int(input_data['ca'])
            input_data['thal'] = int(input_data['thal'])


            # 3. Create DataFrame
            input_df = pd.DataFrame([input_data], columns=feature_names)

            # 4. Scale
            input_scaled = scaler.transform(input_df)

            # 5. Convert scaled data back to DataFrame with names
            input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

            # 6. Predict
            prediction = model.predict(input_scaled_df)
            prediction_proba = model.predict_proba(input_scaled_df)

            # 7. Format result
            result = prediction[0]
            prob_no_disease = prediction_proba[0][0]
            prob_disease = prediction_proba[0][1]

            if result == 1:
                context['prediction_text'] = "Prediction: Has Heart Disease"
            else:
                context['prediction_text'] = "Prediction: Does NOT Have Heart Disease"

            context['prediction_proba_text'] = f"Probability (No Disease: {prob_no_disease:.2f}, Disease: {prob_disease:.2f})"

        except Exception as e:
            context['prediction_text'] = f"Error during prediction: {e}"

    
    return render(request, 'heart.html', context)

# code from the flask app
import io
import os
import pickle
import joblib

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



def index(request):
    # Serve the landing page
    return render(request, 'index.html')


def diabetes_view(request):
    return render(request, 'diabetes_prediction_page.html')

def predict_diabetes_view(request):
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