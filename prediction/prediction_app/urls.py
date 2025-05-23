from django.urls import path
from . import views

app_name = 'prediction_app'

urlpatterns = [
    path('',views.index, name="home_page"),
    #path('prediction/',views.prediction_page, name="prediction_page"),
    path('diabetes/',views.diabetes_view, name="diabetes"),
    path('predict_diabetes/',views.predict_diabetes_view, name="predict_diabetes"),
    #path('plotly/',views.plotly_view, name="plotly"),
    #path('prediction_page/',views.predict_view, name="prediction_page"),
    #path('predict_heart/',views.predict_heart_view, name="predict_heart"),
    path('heart/',views.heart_page, name="heart"),
]

