import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, file_path: str):
        # Converte para Path que é mais fácil de trabalhar
        self.file_path = Path(file_path)
        
    def load(self) -> pd.DataFrame:
        # Verifica se o arquivo existe antes de tentar ler
        if not self.file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")
        
        # Usa pandas para ler o CSV de forma fácil
        df = pd.read_csv(self.file_path)
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        return {
            'total_records': len(df),  # Quantas linhas tem
            'columns': list(df.columns),  # Quais são as colunas
            'missing_values': df.isnull().sum().to_dict(),  # Quantos valores faltam em cada coluna
            'dtypes': df.dtypes.to_dict()  # Que tipo de dado cada coluna armazena
        }

