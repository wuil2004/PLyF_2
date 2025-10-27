from django.urls import path
from .views import train_model, training_form


# --- ¡AGREGA ESTA LÍNEA! ---
app_name = 'decision_forest' 
# ---------------------------

urlpatterns = [
    # Nueva ruta para el formulario web (GET)
    path('', training_form, name='training_form'),

    path('train_model/', train_model, name='train_model_api'),
]
