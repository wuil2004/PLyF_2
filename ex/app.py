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
st.set_page_config(layout="wide", page_title="Detector de Phishing Pro")
warnings.filterwarnings("ignore", category=UserWarning)

# --- ESTILOS CSS (Colores Mejorados) ---
st.markdown("""
    <style>
    /* Fondo principal de la app - Gris muy claro */
    .stApp {
        background-color: #00000; /* Un gris aún más claro, casi blanco */
    }
            
    /* Título principal - Azul oscuro para alto contraste */
    h1 {
        color: #212529; /* Un azul muy oscuro, casi negro */
        text-align: center;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1); /* Sombra sutil para el título */
    }
            
    /* Subtítulo/Markdown - Gris oscuro para legibilidad */
    .stMarkdown p {
        text-align: center;
        color: #495057; /* Un gris oscuro */
        font-size: 1.1em;
    }
            
    /* Estilo de las columnas para que parezcan "tarjetas" */
    [data-testid="column"] {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 24px;
        transition: all 0.3s ease;
    }
            
    [data-testid="column"]:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        transform: translateY(-5px);
    }
            
    /* Encabezados de las tarjetas (h2) - Azul más claro para diferenciar */
    h2 {
        color: #007bff; /* Azul primario de Bootstrap */
        border-bottom: 2px solid #e9ecef; /* Separador más suave */
        padding-bottom: 10px;
    }
            
    /* Estilo de la barra lateral - Blanco */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        padding: 10px;
        box-shadow: 2px 0 8px rgba(0,0,0,0.05); /* Sombra para resaltar */
    }

    /* Estilo del botón en la barra lateral - Azul con mejor contraste */
    .stButton > button {
        background-color: #28a745; /* Verde de éxito de Bootstrap */
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
            
    .stButton > button:hover {
        background-color: #218838; /* Verde más oscuro al pasar el ratón */
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
            
    /* Estilo del slider - Verde para coherencia con el botón */
    .stSlider [data-testid="stThumb"] {
        background-color: #28a745; 
    }
    .stSlider [data-testid="stTrack"] {
        background-color: #e2e6ea;
    }
    .stSlider [data-testid="stFill"] {
        background-color: #28a745;
    }

    /* Estilo del spinner */
    .stSpinner > div > div {
        border-top-color: #28a745; /* Verde */
    }
            
    /* Contenedor de métricas - Fondo más suave */
    [data-testid="stMetric"] {
        background-color: #393d42; /* Verde muy claro para la métrica */
        border-radius: 8px;
        padding: 10px;
        border-left: 5px solid #28a745; /* Borde de color */
    }
    </style>
""", unsafe_allow_html=True)

# --- Funciones de Carga y Preprocesamiento (Cacheadas) ---
@st.cache_data
def load_data():
    """Carga el dataset desde el archivo CSV."""
    try:
        df = pd.read_csv('Phishing.csv')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: No se encontró 'Phishing.csv'. Asegúrate de que esté en la misma carpeta.")
        return None

def preprocess_data(df, train_size):
    """Divide y preprocesa los datos."""
    features_for_model = ['domainUrlRatio', 'domainlength', 'URL_Type_obf_Type']
    df_subset = df[features_for_model].copy()

    imputer = SimpleImputer(strategy='median')
    df_subset[['domainUrlRatio', 'domainlength']] = imputer.fit_transform(df_subset[['domainUrlRatio', 'domainlength']])
    
    X = df_subset.drop('URL_Type_obf_Type', axis=1)
    y_raw = df_subset['URL_Type_obf_Type']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, label_encoder

# --- Función de Visualización (Mejorada) ---
def plot_decision_boundary(clf, X, y, ax, title):
    """Dibuja el límite de decisión con un estilo mejorado."""
    X_values = X.values
    
    # Colores más estéticos para los puntos y fondos de las gráficas
    color_benigno = "#007bff"  # Azul brillante
    color_phishing = "#dc3545" # Rojo brillante
    cmap_background = ListedColormap(['#cfe2ff', '#f8d7da']) # Azul claro, Rojo claro

    mins = X_values.min(axis=0) - 0.1
    maxs = X_values.max(axis=0) + 0.1
    
    x1, x2 = np.meshgrid(np.linspace(mins[0], maxs[0], 200), np.linspace(mins[1], maxs[1], 200))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    
    y_pred = clf.predict(X_new).reshape(x1.shape)
    
    ax.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap_background)

    # Dibujar puntos de datos con nuevos estilos
    ax.plot(X_values[:, 0][y==0], X_values[:, 1][y==0], "o", c=color_benigno, label='Benigno', markersize=6, alpha=0.7, markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(X_values[:, 0][y==1], X_values[:, 1][y==1], "X", c=color_phishing, label='Phishing', markersize=7, alpha=0.7, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_title(title, fontsize=16, fontweight='bold', color='#343a40') # Color de título de gráfica
    ax.set_xlabel('domainUrlRatio', fontsize=12, color='#495057')
    ax.set_ylabel('domainlength', fontsize=12, color='#495057')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6) # Reintroducir grid
    plt.tight_layout() # Ajustar el diseño para evitar recortes

# --- Interfaz de Usuario de Streamlit ---

# --- TÍTULO PRINCIPAL ---
st.title("⚡ Comparador de Modelos de Detección de Phishing ⚡")
st.markdown("<p>Usa el panel de control para entrenar los modelos y comparar sus límites de decisión.</p>", unsafe_allow_html=True)
st.write("") # Espacio

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.title("⚙️ Panel de Control")
st.sidebar.markdown("Ajusta los parámetros de entrenamiento:")

df = load_data()

if df is not None:
    train_size_percentage = st.sidebar.slider(
        "Porcentaje del dataset para entrenamiento:",
        min_value=10,
        max_value=100,
        value=60,
        step=10,
        format="%d%%"
    )
    train_size_float = train_size_percentage / 100.0

    st.sidebar.markdown("---")
    
    # Botón para iniciar el entrenamiento
    if st.sidebar.button(f"🚀 Entrenar modelos ({train_size_percentage}%)"):
        
        # 1. Preprocesar los datos
        with st.spinner("Preprocesando datos y entrenando modelos..."):
            X_train, X_test, y_train, y_test, le = preprocess_data(df, train_size_float)
            
            # --- Modelo SVM ---
            svm_pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('svm_clf', SVC(kernel='linear', C=1))
            ])
            svm_pipeline.fit(X_train, y_train)
            y_pred_svm = svm_pipeline.predict(X_test)
            phishing_label_encoded = le.transform(['phishing'])[0]
            f1_svm = f1_score(y_test, y_pred_svm, pos_label=phishing_label_encoded)
            
            # --- Modelo Árbol de Decisión ---
            dt_pipeline = Pipeline([
                ('tree_clf', DecisionTreeClassifier(max_depth=20, random_state=42))
            ])
            dt_pipeline.fit(X_train, y_train)
            y_pred_dt = dt_pipeline.predict(X_test)
            f1_dt = f1_score(y_test, y_pred_dt, pos_label=phishing_label_encoded)

        st.success(f"¡Modelos entrenados! {len(X_train)} muestras de entrenamiento, {len(X_test)} muestras de prueba.")
        st.write("") # Espacio

        # --- CONTENEDORES DE RESULTADOS ---
        col1, col2 = st.columns(2)

        # --- Tarjeta SVM ---
        with col1:
            st.header("🤖 Máquina de Soporte Vectorial (SVM)")
            st.metric(label="F1-Score (Phishing)", value=f"{f1_svm:.4f}")
            st.markdown("##### Límite de Decisión (Datos de Entrenamiento)")
            
            with st.spinner("Generando gráfico SVM..."):
                plt.style.use('seaborn-v0_8-darkgrid') # Estilo de gráfica
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_decision_boundary(svm_pipeline, X_train, y_train, ax, "SVM (Datos de Entrenamiento)")
                st.pyplot(fig)

        # --- Tarjeta Árbol de Decisión ---
        with col2:
            st.header("🌳 Árbol de Decisión")
            st.metric(label="F1-Score (Phishing)", value=f"{f1_dt:.4f}")
            st.markdown("##### Límite de Decisión (Datos de Entrenamiento)")

            with st.spinner("Generando gráfico Árbol..."):
                plt.style.use('seaborn-v0_8-darkgrid') # Estilo de gráfica
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_decision_boundary(dt_pipeline, X_train, y_train, ax, "Árbol de Decisión (Datos de Entrenamiento)")
                st.pyplot(fig)
    
    else:
        # Mensaje de bienvenida si no se ha presionado el botón
        st.info("👋 ¡Bienvenido! Ajusta los parámetros en la barra lateral y presiona 'Entrenar' para ver la magia.")

else:
    st.warning("No se pudo cargar 'Phishing.csv'. La aplicación no puede continuar.")