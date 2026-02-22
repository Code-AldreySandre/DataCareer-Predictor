import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import os

def split_data(df, target_col='cargo'):
    """
    Implementa o Hold-out estratificado 70%-30% exigido pelo roteiro. 
    A estratificação garante que a proporção das classes (cargos) seja mantida.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Técnica 1: Random Forest. [cite: 43]
    Justificativa: Excelente para lidar com o desbalanceamento de classes 
    e variáveis categóricas após o encoding. [cite: 46]
    """
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        class_weight='balanced' # Lida com o desbalanceamento identificado na EDA
    )
    model.fit(X_train, y_train)
    return model

def train_mlp_neural_network(X_train, y_train):
    """
    Técnica 2: Redes Neurais Artificiais 
    Justificativa: Capaz de aprender relações não-lineares complexas entre 
    ferramentas e cargos. [cite: 46]
    """
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50), # Duas camadas ocultas para profundidade
        activation='relu', 
        solver='adam', 
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Etapa de Avaliação: Extrai acurácia e matriz de confusão. 
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, cm

def save_model(model, filename):
    """Salva o modelo treinado para uso posterior."""
    path = os.path.join(os.path.dirname(__file__), '../models', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path) 