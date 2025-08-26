# Relatório Final – Análise e Modelo de Árvores de Decisão na Base F1

## 1. Introdução
O objetivo deste projeto foi construir um modelo de **árvore de decisão** capaz de classificar a nacionalidade de pilotos da Fórmula 1 com base em suas estatísticas de desempenho. Utilizamos dados históricos da base `drivers.csv`, disponível no Kaggle (1950-2020), e aplicamos técnicas de pré-processamento, normalização e modelagem supervisionada.

---

## 2. Tecnologias Utilizadas
- **Python 3.11** – linguagem principal para manipulação de dados e modelagem.  
- **Pandas** – carregamento e manipulação do dataset.  
- **scikit-learn** – construção do modelo de árvore de decisão, pré-processamento e divisão dos dados.  
- **Matplotlib** – visualização da árvore de decisão.  
- **OpenDatasets** – download do dataset diretamente do Kaggle.  

---

## 3. Etapas do Processo

### 3.1 Aquisição do Dataset
O dataset foi obtido do Kaggle por meio da URL:  
[Formula 1 World Championship 1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)  

O arquivo utilizado foi `drivers.csv`, contendo informações de pilotos, temporadas, vitórias, poles, pontos e nacionalidades.

---

### 3.2 Pré-processamento

**Tratamento de valores ausentes:**  
- Colunas numéricas preenchidas com a média.  
- Colunas categóricas preenchidas com a moda.  

**Normalização:**  
- Todas as colunas numéricas normalizadas para o intervalo [0,1] usando `MinMaxScaler`.  

**Filtragem de nacionalidades:**  
- Apenas nacionalidades com pelo menos 2 pilotos foram consideradas.

---

### 3.3 Preparação de Features e Target
- **Features (X):** todas as colunas numéricas e categóricas transformadas em variáveis dummy (`get_dummies`).  
- **Target (y):** coluna `nationality`.  

---

### 3.4 Divisão do Dataset
O conjunto de dados foi dividido em:  
- **Treino:** 80%  
- **Teste:** 20%  

Estratificado por nacionalidade para garantir representatividade.

---

### 3.5 Modelagem
- **Modelo:** Decision Tree Classifier (`DecisionTreeClassifier`)  
- **Hiperparâmetros:** `max_depth=5` e `random_state=42`  

---

### 3.6 Visualização da Árvore
A árvore de decisão foi gerada utilizando **Matplotlib**, com:  
- Nó preenchido (`filled=True`) para facilitar interpretação.  
- Bordas arredondadas (`rounded=True`).  
- `figsize=(40,20)` para melhor visualização.  

A árvore foi exportada em **SVG**, pronta para inclusão em relatórios ou documentos HTML.

---

## 4. Pontos sobre a Árvore (Decision Tree)
- **Nós de decisão:** Cada nó representa uma condição sobre uma feature, como “wins ≤ 0.35”. Permite ver quais variáveis são mais importantes para classificar a nacionalidade.  
- **Folhas:** Mostram a classe final predita (nacionalidade) para o caminho percorrido. Cores indicam a classe predominante.  
- **Importância das features:** As primeiras divisões (raiz da árvore) indicam as features mais relevantes, por exemplo: vitórias, pontos ou poles.  
- **Perfis de pilotos:** Cada caminho da raiz até uma folha define um perfil típico de piloto daquela nacionalidade.  
- **Distribuição de classes:** Nós totalmente coloridos representam folhas com classe predominante, enquanto nós mistos indicam áreas em que a classificação é menos clara.  
- **Insights possíveis:**  
  - Identificar padrões históricos de desempenho por nacionalidade.  
  - Avaliar a relação entre vitórias, poles, pontos e nacionalidade.  
  - Auxiliar na análise exploratória e decisões estratégicas com base no histórico de pilotos.  

---

## 5. Resultados
- A árvore gerada permitiu observar quais características foram mais relevantes na classificação da nacionalidade dos pilotos.  
- Com `max_depth=5`, a árvore ficou **interpretável**, evitando complexidade excessiva.  

---

## 6. Possíveis Melhorias
1. Criar novas métricas, como razão de vitórias por corridas disputadas.  
2. Ajustar hiperparâmetros do modelo para melhorar acurácia.  
3. Testar outros modelos, como Random Forest ou Gradient Boosting.  
4. Aplicar balanceamento de classes (ex.: SMOTE) para nacionalidades com poucos pilotos.  

---

## 7. Conclusão
O projeto demonstrou o processo completo de **pré-processamento, modelagem e visualização** de dados usando Python e ferramentas de ciência de dados. A árvore de decisão fornece insights visuais sobre como características dos pilotos influenciam na nacionalidade, sendo útil tanto para análise exploratória quanto para classificações automáticas.


```python exec="on" html="1"
--8<-- ".\docs\Decision Tree\avaliação.py"
```
