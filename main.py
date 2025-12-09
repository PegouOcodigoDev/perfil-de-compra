# Importações necessárias para o funcionamento do sistema
import sys
from pathlib import Path
import numpy as np

# Configura o caminho para encontrar os módulos do projeto
sys.path.append(str(Path(__file__).parent))

# Importa as classes principais que fazem o trabalho pesado
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.cluster_engine import ClusterEngine
from src.dashboard_app import DashboardApp


def normalize_cluster_labels(labels: np.ndarray) -> np.ndarray:
    # Pega todos os valores únicos de clusters que existem
    unique_labels = np.unique(labels)
    # Cria uma cópia para não alterar os originais
    normalized_labels = labels.copy()
    
    # Vamos criar um mapeamento: cluster antigo -> cluster novo
    label_mapping = {}
    # Começamos a numerar do 1
    new_label = 1
    
    # Pega só os clusters válidos (excluindo -1 que é usado para outliers)
    valid_labels = unique_labels[unique_labels != -1]
    
    # Para cada cluster válido, renumeramos do 1 em diante
    for old_label in sorted(valid_labels):
        label_mapping[old_label] = new_label
        # Substitui todos os valores do cluster antigo pelo novo número
        normalized_labels[labels == old_label] = new_label
        new_label += 1
    
    # Se tiver outliers (-1), eles viram o último número disponível
    noise_mask = labels == -1
    if noise_mask.any():
        normalized_labels[noise_mask] = new_label
    
    return normalized_labels


def main():
    print("Carregando dados para o dashboard...")
    
    # Primeiro passo: carrega o arquivo CSV com os dados de compras
    loader = DataLoader('data/amazon.csv')
    df_raw = loader.load()
    
    # Segundo passo: transforma os dados brutos em perfis de consumidores
    # Agrupa todas as compras por usuário e calcula médias, totais, etc.
    preprocessor = DataPreprocessor()
    df_profiles = preprocessor.create_user_profiles(df_raw)
    # Prepara os dados no formato que os algoritmos de ML precisam (matriz numérica normalizada)
    X_scaled, _ = preprocessor.prepare_features(df_profiles)
    
    print("Aplicando algoritmos de clustering...")
    print(f"  Dados: {X_scaled.shape[0]} amostras, {X_scaled.shape[1]} features")
    
    # Terceiro passo: aplica K-means para agrupar os consumidores
    # O algoritmo encontra automaticamente o melhor número de grupos
    engine_kmeans = ClusterEngine()
    labels_kmeans_raw, params_kmeans = engine_kmeans.apply_kmeans(X_scaled)
    # Calcula métricas para saber quão bom foi o agrupamento
    metrics_kmeans = engine_kmeans.calculate_metrics(X_scaled, labels_kmeans_raw)
    # Normaliza os rótulos para começarem do 1
    labels_kmeans = normalize_cluster_labels(labels_kmeans_raw)
    
    print(f"  K-means: {params_kmeans['n_clusters']} clusters")
    print(f"    Silhouette: {metrics_kmeans.get('silhouette_score', 0):.3f}")
    print(f"    Distribuição: {dict(zip(*np.unique(labels_kmeans, return_counts=True)))}")
    
    # Quarto passo: aplica DBSCAN para comparar com K-means
    # DBSCAN é um algoritmo diferente que encontra grupos de forma diferente
    engine_dbscan = ClusterEngine()
    labels_dbscan_raw, params_dbscan = engine_dbscan.apply_dbscan(X_scaled)
    # Calcula as métricas do DBSCAN também
    metrics_dbscan = engine_dbscan.calculate_metrics(X_scaled, labels_dbscan_raw)
    # Normaliza os rótulos
    labels_dbscan = normalize_cluster_labels(labels_dbscan_raw)
    
    print(f"  DBSCAN: {len(np.unique(labels_dbscan))} clusters")
    print(f"    Parâmetros: eps={params_dbscan['eps']:.2f}, min_samples={params_dbscan['min_samples']}")
    print(f"    Silhouette: {metrics_dbscan.get('silhouette_score', 0):.3f}")
    unique_dbscan, counts_dbscan = np.unique(labels_dbscan, return_counts=True)
    print(f"    Distribuição: {dict(zip(unique_dbscan, counts_dbscan))}")
    
    # Quinto passo: adiciona os rótulos de cluster de volta aos perfis
    # Agora cada consumidor sabe em qual grupo ele está
    df_kmeans = df_profiles.copy()
    df_kmeans['cluster'] = labels_kmeans
    
    df_dbscan = df_profiles.copy()
    df_dbscan['cluster'] = labels_dbscan
    
    print("Iniciando dashboard...")
    
    # Sexto passo: cria e inicia o dashboard web interativo
    # O dashboard permite visualizar os resultados de forma bonita e explorar os dados
    dashboard = DashboardApp(
        df_kmeans=df_kmeans,
        df_dbscan=df_dbscan,
        metrics_kmeans=metrics_kmeans,
        metrics_dbscan=metrics_dbscan
    )
    
    print("\n" + "="*70)
    print("Dashboard iniciado com sucesso!")
    print("Acesse: http://localhost:8050")
    print("="*70 + "\n")
    
    # Roda o servidor web que fica esperando requisições
    # debug=True mostra erros detalhados durante desenvolvimento
    # host='0.0.0.0' permite acessar de outras máquinas na rede
    dashboard.run(debug=True, host='0.0.0.0', port=8050)


if __name__ == '__main__':
    main()
