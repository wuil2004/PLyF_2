import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# --- Configuración de Matplotlib para no usar GUI en el servidor ---
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# --- Funciones de Lógica ---

def plot_decision_boundary(clf, X, y, title):
    """Genera una imagen del límite de decisión y la devuelve como cadena base64."""
    plt.figure(figsize=(8, 6))
    
    # Escalar datos solo para la visualización para que los ejes sean consistentes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf.fit(X_scaled, y) # Re-entrenar con los datos escalados para el gráfico

    # Crear una malla de puntos para el gráfico
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predecir en cada punto de la malla
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Dibujar el contorno y los puntos de datos
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    
    plt.title(title)
    plt.xlabel('domainUrlRatio (escalado)')
    plt.ylabel('domainlength (escalado)')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Benigno', 'Phishing'])
    
    # Guardar el gráfico en memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def train_and_evaluate_models(train_size_percent):
    """
    Carga, preprocesa y entrena ambos modelos con el dataset de Phishing.
    """
    # 1. Cargar y preprocesar datos
    df = pd.read_csv("datasets/datasets/FinalDataset/Phishing.csv")
    df = df.drop("argPathRatio", axis=1, errors='ignore')
    df = df.fillna(df.median(numeric_only=True))
    
    df['URL_Type_obf_Type'] = df['URL_Type_obf_Type'].replace({'benign': 0, 'phishing': 1})
    
    # Usaremos solo dos características para poder visualizar los límites de decisión
    X = df[['domainUrlRatio', 'domainlength']]
    y = df['URL_Type_obf_Type']

    # 2. Dividir los datos
    train_size_float = train_size_percent / 100.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size_float, random_state=42, stratify=y)
    
    # 3. Entrenar y Evaluar SVM
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel="rbf", C=100, gamma=0.1, random_state=42))
    ])
    svm_pipeline.fit(X_train, y_train)
    y_pred_svm = svm_pipeline.predict(X_test)
    f1_svm = f1_score(y_test, y_pred_svm, pos_label=1)
    
    # 4. Entrenar y Evaluar Árbol de Decisión
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    f1_dt = f1_score(y_test, y_pred_dt, pos_label=1)
    
    # 5. Generar los gráficos de límites de decisión
    # Para los gráficos, usamos el conjunto de prueba para ver cómo se generaliza
    plot_svm_base64 = plot_decision_boundary(SVC(kernel="rbf", C=100, gamma=0.1, random_state=42), X_test, y_test, 'Límite de Decisión - SVM')
    plot_dt_base64 = plot_decision_boundary(DecisionTreeClassifier(max_depth=10, random_state=42), X_test, y_test, 'Límite de Decisión - Árbol')
    
    return f1_svm, f1_dt, plot_svm_base64, plot_dt_base64

# --- Rutas de la API ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    train_size = int(request.form['train_size'])
    
    f1_svm, f1_dt, plot_svm, plot_dt = train_and_evaluate_models(train_size)
    
    return render_template(
        'results.html',
        train_size=train_size,
        f1_svm=round(f1_svm, 4),
        f1_dt=round(f1_dt, 4),
        plot_svm=plot_svm,
        plot_dt=plot_dt
    )

if __name__ == '__main__':
    app.run(debug=True)