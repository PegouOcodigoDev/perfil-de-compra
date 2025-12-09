import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ClusterEngine:
    def __init__(self):
        # Guarda o modelo treinado (para poder usar depois se necessário)
        self.model = None
        # Guarda os rótulos (qual consumidor está em qual grupo)
        self.labels = None
        # Guarda os parâmetros que funcionaram melhor
        self.best_params = {}
        
    def find_optimal_k(self, X: np.ndarray, k_range: Tuple[int, int] = (2, 10)) -> int:
        silhouette_scores = []
        # Testa de k_range[0] até k_range[1], mas não mais que metade dos dados
        # (não faz sentido ter mais grupos que a metade dos consumidores)
        k_values = range(k_range[0], min(k_range[1] + 1, len(X) // 2))
        
        # Testa cada valor de k
        for k in k_values:
            try:
                # Cria um K-means com k grupos
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                # Agrupa os dados
                labels = kmeans.fit_predict(X)
                
                # Precisa ter pelo menos 2 grupos diferentes para calcular métrica
                if len(set(labels)) < 2:
                    silhouette_scores.append(-1)
                    continue
                
                # Calcula o quão bom foi esse agrupamento
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            except Exception:
                # Se der erro, marca como inválido
                silhouette_scores.append(-1)
        
        # Se nenhum funcionou, usa o mínimo possível
        if not silhouette_scores or max(silhouette_scores) < 0:
            return k_range[0]
        
        # Pega o k que deu o maior score (melhor agrupamento)
        optimal_k = k_values[np.argmax(silhouette_scores)]
        return optimal_k
    
    def apply_kmeans(self, X: np.ndarray, n_clusters: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        # Se não especificou quantos grupos, encontra o melhor automaticamente
        if n_clusters is None:
            n_clusters = self.find_optimal_k(X)
        
        # Cria e treina o K-means
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        # Agrupa os dados - retorna um array com o grupo de cada consumidor
        self.labels = self.model.fit_predict(X)
        
        # Guarda os parâmetros usados
        self.best_params = {'n_clusters': n_clusters, 'algorithm': 'kmeans'}
        
        return self.labels, self.best_params
    
    def find_optimal_dbscan_params(self, X: np.ndarray) -> Tuple[float, int]:
        best_score = -1
        best_eps = 0.5
        best_min_samples = 5
        best_n_clusters = 0
        
        # Testa diferentes valores de eps (distância)
        eps_range = np.arange(0.3, 2.0, 0.1)
        # Testa diferentes valores de min_samples (mínimo de vizinhos)
        min_samples_range = [3, 5, 7, 10, 15]
        
        # Testa todas as combinações possíveis
        for eps in eps_range:
            for min_samples in min_samples_range:
                # Cria um DBSCAN com esses parâmetros
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                # Conta quantos grupos formou
                unique_labels = set(labels)
                # -1 significa outlier (não está em nenhum grupo)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels) if len(labels) > 0 else 1.0
                
                # Só considera válido se:
                # - Formou pelo menos 2 grupos
                # - Menos de 50% são outliers
                if n_clusters >= 2 and noise_ratio < 0.5:
                    # Precisamos de pelo menos 2 pontos por grupo para calcular métrica
                    valid_mask = labels != -1
                    if valid_mask.sum() >= n_clusters * 2:
                        try:
                            # Calcula quão bom foi esse agrupamento
                            score = silhouette_score(X[valid_mask], labels[valid_mask])
                            
                            # Se for melhor que os anteriores, guarda
                            if score > best_score or (score == best_score and n_clusters > best_n_clusters):
                                best_score = score
                                best_eps = eps
                                best_min_samples = min_samples
                                best_n_clusters = n_clusters
                        except:
                            # Se der erro, tenta a próxima combinação
                            continue
        
        # Se nenhuma combinação funcionou, usa valores padrão
        if best_score == -1:
            best_eps = 0.8
            best_min_samples = 5
        
        return best_eps, best_min_samples
    
    def apply_dbscan(self, X: np.ndarray, eps: Optional[float] = None, 
                     min_samples: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        # Se não especificou os parâmetros, encontra os melhores automaticamente
        if eps is None or min_samples is None:
            eps, min_samples = self.find_optimal_dbscan_params(X)
        
        # Cria e treina o DBSCAN
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        # Agrupa os dados
        self.labels = self.model.fit_predict(X)
        
        # Guarda os parâmetros usados
        self.best_params = {
            'eps': eps, 
            'min_samples': min_samples, 
            'algorithm': 'dbscan'
        }
        
        return self.labels, self.best_params
    
    def calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        # Conta quantos grupos existem e quantos outliers
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        # Guarda informações básicas
        metrics['n_clusters'] = n_clusters
        metrics['n_noise'] = n_noise
        metrics['noise_ratio'] = n_noise / len(labels) if len(labels) > 0 else 0
        
        # Para calcular métricas, precisamos excluir outliers (-1)
        valid_mask = labels != -1
        n_valid = valid_mask.sum()
        
        # Só calcula métricas se tiver condições mínimas
        if n_clusters > 1 and n_valid >= n_clusters * 2:
            try:
                # Silhouette Score: mede quão similar um ponto é ao seu grupo vs outros grupos
                # Vai de -1 a 1, quanto maior melhor
                metrics['silhouette_score'] = silhouette_score(X[valid_mask], labels[valid_mask])
            except Exception as e:
                metrics['silhouette_score'] = 0.0
            
            try:
                # Davies-Bouldin: mede a separação entre grupos
                # Quanto menor melhor (grupos bem separados)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X[valid_mask], labels[valid_mask])
            except Exception as e:
                metrics['davies_bouldin_score'] = float('inf')
            
            try:
                # Calinski-Harabasz: razão entre separação de grupos e coesão dentro deles
                # Quanto maior melhor
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
            except Exception as e:
                metrics['calinski_harabasz_score'] = 0.0
        else:
            # Se não conseguiu agrupar bem, marca métricas como ruins
            metrics['silhouette_score'] = 0.0
            metrics['davies_bouldin_score'] = float('inf')
            metrics['calinski_harabasz_score'] = 0.0
        
        return metrics
