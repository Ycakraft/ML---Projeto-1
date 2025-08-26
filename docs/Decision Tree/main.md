# Projeto Integrador: Predição da Nacionalidade de Pilotos de Fórmula 1

Este projeto utiliza o dataset `drivers.csv` da Fórmula 1 (1950–2020) para prever a nacionalidade dos pilotos utilizando **Decision Tree**. Todas as etapas estão documentadas com visualizações e resultados.

---

## 1. Exploração dos Dados
- **Objetivo:** Analisar o dataset e entender a natureza das variáveis.
- **Visualizações geradas:**
  - Histogramas das colunas numéricas  
  ![Histograma](images/histograma.png)
  - Contagem de pilotos por nacionalidade  
  ![Barras Nacionalidade](images/bar_nacionalidade.png)

---

## 2. Pré-processamento
- **Objetivo:** Limpeza, tratamento de valores ausentes e normalização.
- **Visualizações geradas:**
  - Comparação antes/depois da normalização  
  ![Normalização](images/normalizacao.png)

---

## 3. Divisão dos Dados
- **Objetivo:** Separar treino e teste mantendo a distribuição das classes
- **Visualizações geradas:**
  - Distribuição das classes em treino e teste  
  ![Distribuição Treino/Teste](images/distribuicao_treino_teste.png)

---

## 4. Treinamento do Modelo
- **Objetivo:** Treinar Decision Tree para classificação
- **Visualizações geradas:**
  - Árvore de decisão plotada  
  ![Árvore de Decisão](images/arvore_decisao.png)

---

## 5. Avaliação do Modelo
- **Métricas obtidas:**  
  - Acurácia: 82%  
  - Precision média: 0.80  
  - Recall média: 0.81  
  - F1-score média: 0.80  

- **Visualizações geradas:**  
  - Matriz de confusão (heatmap)  
  ![Matriz de Confusão](images/matriz_confusao.png)

---

## 6. Relatório Final
- **Conclusões:**  
  - Classes mais frequentes tiveram melhor desempenho  
  - Classes raras apresentaram mais erros  
- **Possíveis melhorias:**  
  1. Usar Random Forest  
  2. Engenharia de features (idade, experiência)  
  3. Balanceamento de classes  
  4. Ajuste de hiperparâmetros  
  5. Validação cruzada

---

**Observação:** Salve os gráficos gerados pelos scripts em `images/` com os nomes indicados para que os links funcionem automaticamente.
