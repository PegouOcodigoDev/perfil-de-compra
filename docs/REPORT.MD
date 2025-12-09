# Relatório Técnico – Agrupamento de Clientes por Perfil de Compra

## 1. Introdução
Este projeto tem como objetivo identificar *perfis de consumidores* a partir da análise de carrinhos de compras, utilizando técnicas de *aprendizado não supervisionado*. A identificação de clusters permite compreender comportamentos de compra, segmentar marketing e apoiar decisões estratégicas.

O dataset utilizado contém *carrinhos de compras fictícios* (ou dados reais provenientes do Kaggle), incluindo variáveis como ticket médio, número de itens, diversidade de categorias e frequência de compras.

---

## 2. Definição do Problema
Dado um conjunto de dados sem rótulos sobre o comportamento de compra dos clientes, deseja-se:

- Descobrir *grupos naturais* de consumidores.
- Encontrar padrões similares entre hábitos de compra.
- Segmentar clientes com base em características compartilhadas.

---

## 3. Técnicas de Agrupamento Utilizadas

### 3.1. K-means
Algoritmo baseado em distância que particiona os dados em k grupos distintos.

*Características:*
- Requer escolher o número de clusters (k).
- É eficiente, rápido e escalável.
- Funciona bem quando os dados estão normalizados.
- Sensível à presença de outliers.

*Vantagens:*
- Fácil interpretação.
- Resultados consistentes com dados uniformemente distribuídos.

---

### 3.2. DBSCAN
Algoritmo baseado em densidade que agrupa pontos que estão próximos uns dos outros.

*Características:*
- Não exige o número de clusters como parâmetro.
- Identifica outliers naturalmente.
- Funciona bem com clusters de formas irregulares.
- Depende de parâmetros como eps e min_samples.

*Vantagens:*
- Bom para detectar ruído.
- Útil em distribuições não uniformes.

---

## 4. Preparação dos Dados
As seguintes etapas foram realizadas:

- Limpeza e tratamento de valores ausentes.
- Padronização dos atributos numéricos (Z-score).
- Seleção das variáveis mais relevantes para clusterização.
- Redução de dimensionalidade com PCA para visualização dos clusters.

---

## 5. Qual Algoritmo é Melhor para Este Caso?
Com base nos experimentos realizados:

### O K-means apresentou desempenho superior.

*Motivos:*
- Dados padronizados favorecem o uso de distância euclidiana.
- A distribuição dos dados era relativamente uniforme.
- Métricas como silhouette score foram mais altas no K-means.
- Os clusters gerados foram mais interpretáveis e estáveis.

O DBSCAN foi útil para identificar ruído, mas não produziu uma segmentação clara devido à densidade pouco variada do dataset.

*Conclusão:*
> Para este projeto, o algoritmo mais adequado para identificar perfis de compra foi o *K-means*, pelos resultados mais robustos e consistentes.

---

## 6. Conclusão
O estudo mostrou que técnicas de clusterização são ferramentas eficazes para identificar perfis de consumidores. Entre os algoritmos testados, o *K-means* demonstrou melhor desempenho para este dataset, fornecendo clusters bem definidos e insights úteis.

O DBSCAN, embora eficiente para detecção de outliers, não gerou agrupamentos adequados neste cenário devido à baixa variação de densidade.
---
