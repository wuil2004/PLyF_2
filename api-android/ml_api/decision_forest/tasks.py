# decision_forest/tasks.py

import pandas as pd
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient

from celery import shared_task
from django.conf import settings

# Número de características más importantes que queremos mantener
TOP_FEATURES_COUNT = 20 

@shared_task
def train_and_save_model(n_samples):
    """
    Ejecuta el entrenamiento del modelo de forma asíncrona,
    optimizando la velocidad mediante la selección de características.
    """
    try:
        # Lógica de conexión a Mongo
        client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client["malware_db"]
        collection = db["network_flows"]
        
        data = list(collection.find())
        if not data:
            return {"status": "FAILED", "error": "La colección de MongoDB está vacía."}

        df = pd.DataFrame(data).drop('_id', axis=1)
        if 'label' not in df.columns:
            return {"status": "FAILED", "error": "La columna 'label' no se encontró."}
        
        df.replace([np.inf, -np.inf, np.nan, -1], 0, inplace=True)
            
        n_samples_to_use = min(n_samples, len(df))
        df = df.sample(n=n_samples_to_use, random_state=42)

        # 1. Separación inicial para Feature Selection
        X_full = df.drop('label', axis=1)
        y = df['label']
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names_list = le.classes_.tolist()

        # --- FASE DE OPTIMIZACIÓN: ENCONTRAR MEJORES FEATURES ---
        # 2. Entrenar un modelo rápido para obtener la importancia
        fs_model = RandomForestClassifier(random_state=42, n_estimators=5) # n_estimators bajo para velocidad
        fs_model.fit(X_full, y_encoded)
        
        # 3. Seleccionar las N características más importantes
        feature_importances = pd.Series(fs_model.feature_importances_, index=X_full.columns)
        top_features = feature_importances.nlargest(TOP_FEATURES_COUNT).index.tolist()
        
        # 4. Crear el nuevo dataset optimizado
        X_optimized = X_full[top_features]
        # --------------------------------------------------------

        # 5. Entrenamiento Final con Dataset Optimizado
        X_train, X_test, y_train, y_test = train_test_split(X_optimized, y_encoded, test_size=0.2, random_state=42)
        
        # Usamos 10 árboles, pero ahora solo con 20 columnas (¡mucho más rápido!)
        model = RandomForestClassifier(random_state=42, n_estimators=10) 
        model.fit(X_train, y_train)

        # 6. Predicción y F1-Score
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # 7. Generación de Gráficas (usa X_optimized.columns)

        # Gráfica de Importancia (usando las 20 features seleccionadas)
        importances = model.feature_importances_
        plt.figure(figsize=(10,5))
        plt.barh(np.array(X_optimized.columns), importances) 
        plt.title(f"Importancia de Características (Top {TOP_FEATURES_COUNT} Usadas)")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        feature_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close() 

        # Diagrama de un Árbol Individual (usando las 20 features seleccionadas)
        fig, ax = plt.subplots(figsize=(12, 8))
        tree.plot_tree(
            model.estimators_[0], 
            filled=True, 
            feature_names=X_optimized.columns.tolist(), 
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
        
        # 8. Retorno de Resultados
        return {
            "status": "COMPLETED",
            "f1_score": f1,
            "feature_importance_plot": feature_img,
            "decision_tree_diagram": tree_img,
            "n_samples": n_samples_to_use
        }
    
    except Exception as e:
        return {"status": "FAILED", "error": f"Error en la tarea de Celery: {str(e)}"}