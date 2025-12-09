import pandas as pd
import numpy as np
import re
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self):
        # O scaler normaliza os dados (coloca tudo na mesma escala)
        # Isso é importante porque preços podem ser 1000 e descontos 50
        # Precisamos normalizar para comparar coisas diferentes
        self.scaler = None
        # Guarda o nome das colunas que estamos usando como features
        self.feature_names = []
        
    def clean_price(self, price_str) -> float:
        # Se for vazio ou inválido, retorna zero
        if pd.isna(price_str):
            return 0.0
        # Remove símbolos de moeda e vírgulas que separam milhares
        price_str = str(price_str).replace('₹', '').replace(',', '').strip()
        # Procura por números (pode ter ponto decimal)
        numbers = re.findall(r'\d+\.?\d*', price_str)
        # Pega o primeiro número encontrado, se houver
        return float(numbers[0]) if numbers else 0.0
    
    def clean_discount(self, discount_str) -> float:
        # Se for vazio, retorna zero
        if pd.isna(discount_str):
            return 0.0
        # Extrai só os números
        numbers = re.findall(r'\d+\.?\d*', str(discount_str))
        return float(numbers[0]) if numbers else 0.0
    
    def extract_main_category(self, category_str) -> str:
        # Se for vazio, retorna "Unknown"
        if pd.isna(category_str):
            return 'Unknown'
        # Divide pela barra vertical e pega a primeira parte
        parts = str(category_str).split('|')
        return parts[0] if parts else 'Unknown'
    
    def create_user_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        # Faz uma cópia para não alterar o original
        df = df.copy()
        
        # Limpa os preços e descontos que podem vir sujos
        df['discounted_price_clean'] = df['discounted_price'].apply(self.clean_price)
        df['actual_price_clean'] = df['actual_price'].apply(self.clean_price)
        df['discount_pct'] = df['discount_percentage'].apply(self.clean_discount)
        
        # Extrai a categoria principal (primeira parte antes do |)
        df['main_category'] = df['category'].apply(self.extract_main_category)
        # Converte rating para número (pode ter texto misturado)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Limpa os IDs de usuário
        df['user_id'] = df['user_id'].fillna('Unknown')
        # Se tiver múltiplos IDs separados por vírgula, pega só o primeiro
        df['user_id'] = df['user_id'].astype(str).str.split(',').str[0]
        
        # AQUI É O PULO DO GATO: agrupa tudo por usuário
        # Para cada usuário, calcula estatísticas de todas suas compras
        user_profile = df.groupby('user_id').agg({
            'product_id': 'count',  # Quantos produtos comprou
            'discounted_price_clean': ['mean', 'sum', 'std', 'min', 'max'],  # Estatísticas de preço
            'discount_pct': ['mean', 'std', 'max'],  # Estatísticas de desconto
            'rating': ['mean', 'std', 'count'],  # Estatísticas de avaliação
            'main_category': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'  # Categoria mais frequente
        }).reset_index()
        
        # Renomeia as colunas para ficarem mais claras
        user_profile.columns = [
            'user_id', 'total_products', 
            'avg_price', 'total_spent', 'price_std', 'min_price', 'max_price',
            'avg_discount', 'discount_std', 'max_discount',
            'avg_rating', 'rating_std', 'rating_count',
            'favorite_category'
        ]
        
        # Preenche valores faltando com zero
        user_profile = user_profile.fillna(0)
        
        # Cria algumas features adicionais interessantes
        user_profile['price_range'] = user_profile['max_price'] - user_profile['min_price']
        
        # Cria flags binárias (0 ou 1) para características importantes
        # 1 se o usuário busca descontos mais que a média
        user_profile['discount_seeker'] = (user_profile['avg_discount'] > user_profile['avg_discount'].median()).astype(int)
        # 1 se gasta mais que 75% dos outros
        user_profile['high_spender'] = (user_profile['total_spent'] > user_profile['total_spent'].quantile(0.75)).astype(int)
        # 1 se compra mais frequentemente que a média
        user_profile['frequent_buyer'] = (user_profile['total_products'] > user_profile['total_products'].median()).astype(int)
        
        # Cria colunas para cada categoria (one-hot encoding)
        # Isso transforma categorias em números (0 ou 1 para cada categoria)
        category_dummies = pd.get_dummies(
            df.groupby('user_id')['main_category'].apply(list).apply(pd.Series).stack().reset_index(level=1, drop=True),
            prefix='cat'
        )
        category_counts = category_dummies.groupby(category_dummies.index).sum()
        
        # Junta as informações de categoria com o perfil do usuário
        user_profile = user_profile.merge(category_counts, left_on='user_id', right_index=True, how='left')
        user_profile = user_profile.fillna(0)
        
        return user_profile
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        # Features obrigatórias que sempre usamos
        required_features = ['avg_price', 'avg_discount']
        
        # Verifica se as features obrigatórias existem
        for feature in required_features:
            if feature not in df.columns:
                raise ValueError(f"Feature obrigatória '{feature}' não encontrada nos dados")
        
        df_work = df.copy()
        # Garante que preço e desconto são números
        df_work['avg_price'] = pd.to_numeric(df_work['avg_price'], errors='coerce').fillna(0)
        df_work['avg_discount'] = pd.to_numeric(df_work['avg_discount'], errors='coerce').fillna(0)
        
        # Se tiver categoria favorita, converte para números (one-hot encoding)
        # Cada categoria vira uma coluna com 0 ou 1
        if 'favorite_category' in df_work.columns:
            category_dummies = pd.get_dummies(df_work['favorite_category'], prefix='cat')
            df_work = pd.concat([df_work[required_features], category_dummies], axis=1)
            feature_names = required_features + list(category_dummies.columns)
        else:
            # Se não tiver categoria, usa só preço e desconto
            df_work = df_work[required_features]
            feature_names = required_features
        
        # Converte para matriz numpy (array de arrays)
        X = df_work.fillna(0).values
        
        # Remove features que não variam (são sempre iguais para todos)
        # Isso é importante porque features constantes não ajudam a diferenciar grupos
        variance = np.var(X, axis=0)
        valid_features = variance > 1e-8
        
        if valid_features.sum() == 0:
            raise ValueError("Todas as features têm variância zero")
        
        # Mantém só as features que variam
        X = X[:, valid_features]
        self.feature_names = [feature_names[i] for i in range(len(feature_names)) if valid_features[i]]
        
        # Normaliza os dados para que todos fiquem na mesma escala
        # Por exemplo: preços podem ser 1000 e descontos 50
        # Após normalizar, ambos ficam entre -1 e 1 aproximadamente
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Garante que não tem valores inválidos (infinito ou NaN)
        # Substitui por zero se encontrar algum
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_scaled, self.feature_names
    
    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calcula correlação entre features
        correlation_matrix = df[self.feature_names].corr().abs()
        # Features mais correlacionadas são mais importantes
        importance = correlation_matrix.sum().sort_values(ascending=False)
        
        return pd.DataFrame({
            'feature': importance.index,
            'importance_score': importance.values
        })
