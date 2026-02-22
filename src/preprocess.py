import pandas as pd
import numpy as np
import joblib
import os

def clean_data(df):
    """
    Etapa de Pré-processamento do KDD: Tratamento de ruídos e nulos. [cite: 31, 32]
    """
    df_clean = df.copy()

    # Tratando outliers
    df_clean = df_clean[(df_clean['idade'] >= 16) & (df_clean['idade'] <= 75)]
    
    # Tratando valores nulos
    cols_to_fill = ['linguagens_preferidas', 'bancos_de_dados', 'cloud_preferida', 'formacao']
    df_clean[cols_to_fill] = df_clean[cols_to_fill].fillna('Não Informado/Não utiliza')
    
    # Preenche idade faltante com a mediana 
    df_clean['idade'] = df_clean['idade'].fillna(df_clean['idade'].median())
    
    return df_clean

def transform_experience(df):
    """
    Criação de Atributos: Converte tempo de experiência em valor numérico.  Isso facilita o processamento por Redes Neurais.
    """
    mapping = {
        'Não tenho experiência na área de dados': 0,
        'Menos de 1 ano': 0.5,
        'de 1 a 2 anos': 1.5,
        'de 2 a 3 anos': 2.5,
        'de 3 a 4 anos': 3.5,
        'de 4 a 6 anos': 5.0,
        'de 6 a 10 anos': 8.0,
        'Mais de 10 anos': 12.0
    }
    df['experiencia_num'] = df['tempo_experiencia_dados'].map(mapping).fillna(0)
    return df

def process_multilabel_features(df):
    """
    Lógica avançada: Transforma strings como 'Python, R, SQL' em colunas binárias.
    Útil para Redes Neurais captarem o conhecimento em múltiplas ferramentas. [cite: 37]
    """
    # para Linguagens
    languages = df['linguagens_preferidas'].str.get_dummies(sep=', ')
    # Pra não confundir com outras coluna
    languages = languages.add_prefix('lang_')
    
    # Concatenamos e removemos a original
    df = pd.concat([df, languages], axis=1)
    return df

def encode_and_format(df):
    """
    Etapa de Formatação: One-Hot Encoding para variáveis categóricas de escolha única. [cite: 36, 37]
    """
    # Atributos sugeridos no roteiro 
    categorical_features = [
        'genero', 'etnia', 'pcd', 'vive_no_brasil', 
        'nivel_ensino', 'formacao', 'cloud_preferida'
    ]
    
    # Evita Dummy Variable Trap
    df_final = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Remoção de colunas de texto puro que não serão usadas pelo modelo
    cols_to_drop = ['estado_moradia', 'tempo_experiencia_dados', 
                    'linguagens_preferidas', 'bancos_de_dados']
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])
    
    return df_final

def save_processed_data(df, file_name='dataset_final.pkl'):
    """
    Serializa o dado tratado para garantir persistência e performance.
    """
    output_path = os.path.join(os.path.dirname(__file__), '../data/processed', file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(df, output_path)
    print(f"Dados salvos com sucesso em: {output_path}")