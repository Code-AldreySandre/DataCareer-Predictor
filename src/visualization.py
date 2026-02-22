import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import os

def plot_classification_metrics(model, X_test, y_test, model_name):
    """
    Exibe e salva Precision, Recall e F1-Score para cada cargo. 
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_report.iloc[:-3, :-1], annot=True, cmap='RdYlGn', fmt='.2f')
    plt.title(f'Métricas por Classe - {model_name}')
    
    # Salvando a imagem
    path = f'../reports/figures/metrics_{model_name.lower().replace(" ", "_")}.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Relatório de métricas salvo em: {path}")

def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Gera e salva a matriz de confusão visual. [cite: 47]
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_)
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Salvando a imagem
    path = f'../reports/figures/cm_{model_name.lower().replace(" ", "_")}.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Matriz de confusão salva em: {path}")

def compare_performances(acc_rf, acc_mlp):
    """
    Gera e salva o gráfico de barras comparando a acurácia geral. [cite: 49, 50]
    """
    models = ['Random Forest', 'MLP (Rede Neural)']
    accuracies = [acc_rf, acc_mlp]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=accuracies, palette='magma')
    plt.ylim(0, 1)
    plt.title('Comparação de Acurácia: Random Forest vs MLP')
    plt.ylabel('Acurácia (%)')
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # Salvando a imagem
    path = '../reports/figures/comparativo_acuracia.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Comparativo salvo em: {path}")