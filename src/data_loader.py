import pandas as pd
import os

def load_raw_data(file_name='sods.csv'):
    """
    Carrega o dataset original da pasta raw.
    """
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../data/raw', file_name)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Sucesso: Dataset carregado com {df.shape[0]} instâncias e {df.shape[1]} atributos.")
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo {file_name} não foi encontrado em data/raw/.")
        return None

def get_labeled_data(df):
    """
    Etapa de Seleção do KDD: Filtra apenas instâncias que possuem o target (cargo).
    Para aprendizado supervisionado, instâncias sem rótulo não podem 
    ser usadas no treino/teste dos classificadores.
    """
    if df is not None:
        initial_count = len(df)
        
        df_labeled = df.dropna(subset=['cargo']).copy()
        final_count = len(df_labeled)
        
        print(f"Seleção KDD: {initial_count - final_count} instâncias removidas por falta de rótulo.")
        print(f"Total para modelagem: {final_count} instâncias.")
        
        return df_labeled
    return None