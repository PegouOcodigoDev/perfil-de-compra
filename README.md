# 🎯 Sobre o Projeto

Este projeto foi desenvolvido para analisar dados de compras de consumidores e identificar padrões de comportamento de compra. Através de algoritmos de machine learning, o sistema agrupa automaticamente consumidores que apresentam características semelhantes, como:

- **Preferência por faixa de preço**: Consumidores que compram produtos em faixas de preço similares
- **Sensibilidade a descontos**: Agrupa consumidores com comportamento similar em relação a ofertas e promoções
- **Interesse por categorias**: Identifica grupos que preferem as mesmas categorias de produtos

Com esses agrupamentos, é possível entender melhor o perfil de cada grupo de consumidores e criar estratégias mais direcionadas para cada segmento.

## 💡 Por Que Este Projeto Existe?

Em e-commerce e varejo, entender o comportamento dos consumidores é fundamental para:

- **Personalizar recomendações**: Mostrar produtos que realmente interessam a cada grupo
- **Otimizar campanhas de marketing**: Criar mensagens e ofertas específicas para cada perfil
- **Melhorar a experiência do cliente**: Entender o que cada grupo valoriza
- **Aumentar conversão**: Oferecer produtos e preços adequados a cada segmento

Este sistema automatiza esse processo de análise, identificando automaticamente grupos de consumidores semelhantes sem necessidade de análise manual extensiva.

## 🔍 O Que O Sistema Faz?

### 1. Análise de Dados de Compras

O sistema analisa dados históricos de compras, focando em três dimensões principais:

- **Preço médio**: Valor médio dos produtos que cada consumidor compra
- **Desconto médio**: Sensibilidade a ofertas e promoções
- **Categorias preferidas**: Tipos de produtos mais consumidos

### 2. Agrupamento Inteligente

Utilizando dois algoritmos diferentes de machine learning:

- **K-means**: Agrupa consumidores em um número fixo de grupos, encontrando automaticamente o melhor número de clusters
- **DBSCAN**: Identifica grupos baseado em densidade, podendo encontrar padrões mais complexos e identificar outliers

Ambos os métodos são otimizados automaticamente para encontrar a melhor configuração possível.

### 3. Visualização e Exploração

O sistema oferece um dashboard interativo onde você pode:

- **Comparar métodos**: Ver como K-means e DBSCAN agruparam os consumidores
- **Analisar métricas de qualidade**: Entender quão bem os grupos foram formados
- **Explorar perfis**: Ver detalhes dos consumidores em cada grupo
- **Identificar padrões**: Entender características comuns dentro de cada grupo

## 📊 Resultados e Insights

Após executar a análise, você terá acesso a:

### Métricas de Qualidade

O sistema calcula automaticamente métricas que indicam a qualidade do agrupamento:

- **Silhouette Score**: Quão bem separados e coesos estão os grupos (0 a 1, quanto maior melhor)
- **Davies-Bouldin**: Quão distintos são os grupos entre si (quanto menor melhor)
- **Calinski-Harabasz**: Quão bem definidos estão os grupos (quanto maior melhor)
- **Taxa de Ruído**: Porcentagem de consumidores que não se encaixam bem em nenhum grupo (DBSCAN)

### Visualizações

- **Distribuição dos grupos**: Ver quantos consumidores estão em cada grupo
- **Análise de valor**: Comparar preço médio e gasto total por grupo
- **Detalhes por grupo**: Explorar consumidores individuais dentro de cada grupo, ordenados por preço

## 🚀 Como Usar

### Requisitos

- Python 3.8 ou superior
- Arquivo CSV com dados de compras no formato esperado

### Instalação e Configuração

1. **Clone ou baixe o projeto**

2. **Crie um ambiente virtual** (recomendado):
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

### Executando a Análise

1. **Coloque seus dados**: Certifique-se de que o arquivo `data/amazon.csv` contém os dados de compras no formato esperado

2. **Ative o ambiente virtual** (se ainda não estiver ativo):
```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. **Execute o sistema**:
```bash
python main.py
```

O sistema irá:
- Carregar e processar os dados
- Criar perfis de consumidores baseados em preço, desconto e categorias
- Executar os algoritmos de clustering (K-means e DBSCAN)
- Otimizar automaticamente os parâmetros
- Calcular métricas de qualidade
- Preparar os dados para visualização

3. **Acesse o Dashboard**:

Após a execução, o dashboard será iniciado automaticamente. Você verá uma mensagem no terminal indicando que o servidor está rodando. Acesse no navegador:

```
http://localhost:8050
```

4. **Para parar o servidor**:

Pressione `Ctrl + C` no terminal onde o servidor está rodando.

### Navegando pelo Dashboard

1. **Métricas Gerais**: No topo, veja o resumo de cada método com suas métricas de qualidade
2. **Seleção de Método**: Escolha entre K-means ou DBSCAN para visualizar
3. **Gráficos**: Explore distribuições e análises de valor
4. **Tabela de Resumo**: Veja estatísticas agregadas de cada grupo
5. **Detalhes**: Explore os consumidores individuais em cada grupo (máximo de 10 por grupo, ordenados por preço)