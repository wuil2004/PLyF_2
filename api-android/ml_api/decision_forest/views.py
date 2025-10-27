from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import tree
import matplotlib
# CLAVE: Usar 'Agg' para prevenir fallos de GUI en el servidor
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import pandas as pd
import base64
import io
import numpy as np 
from sklearn.preprocessing import LabelEncoder 

from django.conf import settings
from pymongo import MongoClient
from django.shortcuts import render, redirect
from django.http import HttpResponse 


# --- Función para conectar a Mongo ---
def get_mongo_collection():
    """Establece la conexión a MongoDB y retorna la colección de datos."""
    try:
        # Usamos un timeout bajo para la vista sincrónica
        client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') 
        
        db = client["malware_db"]
        collection = db["network_flows"]
        return collection
    except Exception as e:
        # En producción, esta excepción indica un fallo de conexión
        return None


# --- Vista para mostrar el formulario (GET) ---
def training_form(request):
    """Renderiza el formulario y maneja la visualización inicial/resultados."""
    return render(request, 'decision_forest/training_form.html')


# --- Vista para procesar el entrenamiento (POST) ---
def train_model(request):
    """
    Ejecuta el entrenamiento del modelo de forma SINCRÓNICA,
    optimizando la carga de datos con muestreo directo en MongoDB.
    """
    if request.method != 'POST':
        return redirect('decision_forest:training_form') 

    n_samples_str = request.POST.get('n_samples', '100')
    
    try:
        n_samples = int(n_samples_str)
        if n_samples <= 0:
            raise ValueError("El número de muestras debe ser positivo.")

        # 1. Conexión a MongoDB
        collection = get_mongo_collection()
        if collection is None:
            raise ConnectionError("No se pudo establecer conexión con la base de datos MongoDB.")

        # --- OPTIMIZACIÓN CLAVE: Muestreo Directo en MongoDB ---
        # Usamos la pipeline de agregación para que Mongo seleccione las muestras.
        
        # Primero, obtenemos el número total de documentos para evitar pedir más de los que hay
        total_docs = collection.count_documents({})
        n_samples_to_use = min(n_samples, total_docs)

        pipeline = [
            { '$sample': { 'size': n_samples_to_use } }
        ]
        
        # Carga solo los documentos muestreados, no todo el dataset
        data = list(collection.aggregate(pipeline)) 
        
        if not data:
            raise ValueError("No se pudieron muestrear datos de la colección.")
            
        df = pd.DataFrame(data).drop('_id', axis=1)

        if 'label' not in df.columns:
            raise KeyError("La columna 'label' no se encontró en los datos.")

        # 2. Preparación (Ya no necesitamos df.sample() porque Mongo lo hizo)
        # Limpieza de datos (para prevenir errores del modelo con -1, NaN, etc.)
        df.replace([np.inf, -np.inf, np.nan, -1], 0, inplace=True) 

        X = df.drop('label', axis=1)
        y = df['label']
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names_list = le.classes_.tolist()

        # 3. Entrenamiento (usando 5 estimadores optimizados)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42, n_estimators=5) 
        model.fit(X_train, y_train)

        # 4. Predicción y F1-Score
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # 5. Generación y Codificación de Gráficas

        # Gráfica de Importancia
        importances = model.feature_importances_
        top_n = 15
        indices = importances.argsort()[-top_n:]
        
        plt.figure(figsize=(10,5))
        plt.barh(np.array(X.columns)[indices], importances[indices]) 
        plt.title("Importancia de características (Top 15)")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        feature_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Diagrama de un Árbol Individual
        fig, ax = plt.subplots(figsize=(12, 8))
        tree.plot_tree(
            model.estimators_[0], 
            filled=True, 
            feature_names=X.columns.tolist(), 
            max_depth=3, 
            ax=ax, 
            class_names=class_names_list
        )
        plt.title("Diagrama del Primer Árbol de Decisión (Máx. Profundidad 3)")
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        tree_img = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close()
        
        # 6. Retorno de Resultados
        results = {
            "f1_score": f1,
            "feature_importance_plot": feature_img,
            "decision_tree_diagram": tree_img,
        }

        return render(request, 'decision_forest/training_form.html', {
            'results': results,
            'n_samples': n_samples_to_use,
            'success_message': f'¡Modelo entrenado con éxito! Se utilizaron {n_samples_to_use} muestras (Muestreo en Mongo).'
        })
    
    except Exception as e:
        # 7. Manejo de Errores
        return render(request, 'decision_forest/training_form.html', {
            'error': f'Error al procesar el entrenamiento: {str(e)}',
            'n_samples': n_samples_str 
        })