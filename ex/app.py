import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Detector de Phishing Interactivo")
warnings.filterwarnings("ignore", category=UserWarning)

# --- Funciones de Carga y Preprocesamiento (Cacheadas para eficiencia) ---
@st.cache_data
def load_data():
    """Carga el dataset desde el archivo CSV."""
    try:
        df = pd.read_csv('Phishing.csv')
        # Tratar valores infinitos globalmente al cargar
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo 'phishing.csv'. Asegúrate de que esté en la misma carpeta que 'app.py'.")
        return None

def preprocess_data(df, train_size):
    """Divide y preprocesa los datos según el tamaño de entrenamiento seleccionado."""
    # Seleccionar solo las características relevantes para el modelo y la visualización
    features_for_model = ['domainUrlRatio', 'domainlength', 'URL_Type_obf_Type']
    df_subset = df[features_for_model].copy()

    # Imputar valores faltantes
    imputer = SimpleImputer(strategy='median')
    df_subset[['domainUrlRatio', 'domainlength']] = imputer.fit_transform(df_subset[['domainUrlRatio', 'domainlength']])
    
    X = df_subset.drop('URL_Type_obf_Type', axis=1)
    y_raw = df_subset['URL_Type_obf_Type']

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, label_encoder

# --- Función de Visualización ---
def plot_decision_boundary(clf, X, y, ax, title):
    """Dibuja el límite de decisión para un clasificador dado."""
    X_values = X.values
    
    # Definir límites del gráfico
    mins = X_values.min(axis=0) - 0.1
    maxs = X_values.max(axis=0) + 0.1
    
    # Crear una malla de puntos
    x1, x2 = np.meshgrid(np.linspace(mins[0], maxs[0], 200), np.linspace(mins[1], maxs[1], 200))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    
    # Predecir en la malla
    y_pred = clf.predict(X_new).reshape(x1.shape)
    
    # Dibujar contornos de decisión
    custom_cmap = ListedColormap(['#9898ff', '#fafab0']) # Azul para benigno, Amarillo para phishing
    ax.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)

    # Dibujar los puntos de datos
    ax.plot(X_values[:, 0][y==0], X_values[:, 1][y==0], "bs", label='Benigno')
    ax.plot(X_values[:, 0][y==1], X_values[:, 1][y==1], "g^", label='Phishing')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('domainUrlRatio', fontsize=12)
    ax.set_ylabel('domainlength', fontsize=12)
    ax.legend()

# --- Interfaz de Usuario de Streamlit ---
st.title("🧪 Comparador Interactivo de Modelos de Detección de Phishing")
st.markdown("Usa el slider para seleccionar la cantidad de datos para entrenar los modelos y observa cómo cambian su rendimiento y sus límites de decisión en tiempo real.")

df = load_data()

if df is not None:
    # Slider para seleccionar el tamaño del conjunto de entrenamiento
    train_size_percentage = st.slider(
        "Porcentaje del dataset para entrenamiento:",
        min_value=10,
        max_value=100,
        value=60,
        step=10,
        format="%d%%"
    )
    train_size_float = train_size_percentage / 100.0

    # Botón para iniciar el entrenamiento
    if st.button(f"Entrenar modelos con {train_size_percentage}% de los datos"):
        
        # 1. Preprocesar los datos con el tamaño seleccionado
        with st.spinner("Preprocesando datos..."):
            X_train, X_test, y_train, y_test, le = preprocess_data(df, train_size_float)
        st.success(f"Datos procesados: {len(X_train)} muestras de entrenamiento, {len(X_test)} muestras de prueba.")

        # Contenedores para los resultados
        col1, col2 = st.columns(2)

        # --- Modelo SVM ---
        with col1:
            st.header("Máquina de Soporte Vectorial (SVM)")
            with st.spinner("Entrenando SVM..."):
                # Pipeline para SVM: Escala los datos y luego aplica el clasificador
                svm_pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    ('svm_clf', SVC(kernel='linear', C=1))
                ])
                svm_pipeline.fit(X_train, y_train)
                
                # Evaluación
                y_pred_svm = svm_pipeline.predict(X_test)
                phishing_label_encoded = le.transform(['phishing'])[0]
                f1_svm = f1_score(y_test, y_pred_svm, pos_label=phishing_label_encoded)
                
                st.metric(label="F1-Score (Phishing)", value=f"{f1_svm:.4f}")
                
                # Gráfica
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_decision_boundary(svm_pipeline, X_train, y_train, ax, "Límite de Decisión - SVM")
                st.pyplot(fig)

        # --- Modelo Árbol de Decisión ---
        with col2:
            st.header("Árbol de Decisión")
            with st.spinner("Entrenando Árbol de Decisión..."):
                # Pipeline para Árbol de Decisión (sin escalado)
                dt_pipeline = Pipeline([
                    ('tree_clf', DecisionTreeClassifier(max_depth=20, random_state=42))
                ])
                dt_pipeline.fit(X_train, y_train)

                # Evaluación
                y_pred_dt = dt_pipeline.predict(X_test)
                f1_dt = f1_score(y_test, y_pred_dt, pos_label=phishing_label_encoded)
                
                st.metric(label="F1-Score (Phishing)", value=f"{f1_dt:.4f}")

                # Gráfica
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_decision_boundary(dt_pipeline, X_train, y_train, ax, "Límite de Decisión - Árbol")
                st.pyplot(fig)
