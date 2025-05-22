from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
import joblib
import pandas as pd
import os
from django.conf import settings 
import plotly.offline as opy
import plotly.graph_objs as go
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