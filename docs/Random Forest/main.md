#  Relatório de Análise – Modelos de Classificação Supervisionada

##  Objetivo  
O objetivo deste estudo foi **avaliar o desempenho de diferentes algoritmos de classificação supervisionada** na tarefa de **prever a era dos pilotos de Fórmula 1** com base em variáveis demográficas e históricas.  

Os modelos testados incluem:  
- **Logistic Regression**  
- **Random Forest Classifier**  
- **Gradient Boosting Classifier**  
- **Support Vector Classifier (SVC)**  

A meta principal foi identificar **qual modelo entrega melhor desempenho geral e equilíbrio entre precisão e generalização**, considerando a acurácia e as métricas de *precision*, *recall* e *f1-score*.  

---

##  Conjunto de Dados  
- **Total de registros:** 259 pilotos  
- **Classes Alvo (Eras de Pilotos):**
  -  **Pioneiros (<1960)**  
  -  **Clássicos (60–70s)**  
  -  **Modernos (80–90s)**  
  -  **Contemporâneos (2000+)**  

- **Distribuição das Classes:**  
  - Pioneiros: 200  
  - Clássicos: 35  
  - Modernos: 22  
  - Contemporâneos: 2  

> A base apresenta um **forte desbalanceamento de classes**, com grande predominância dos pilotos “Pioneiros”, o que impacta o aprendizado dos modelos e reduz o desempenho nas classes minoritárias.  

---

##  Pré-Processamento  
1. **Codificação de variáveis categóricas** utilizando *LabelEncoder*  
2. **Padronização numérica** com *StandardScaler*  
3. **Divisão** em *train/test* (80/20)  
4. **Avaliação** via métricas de classificação (*precision, recall, f1-score*) e **acurácia total**  

---

##  Resultados dos Modelos  

| Modelo | Acurácia | Melhor Classe Prevista | Observações |
|:--|:--:|:--:|:--|
| **Logistic Regression** | 0.7722 | 🏁 Pioneiros | Bom desempenho geral, mas falhou nas classes minoritárias. |
| **Random Forest** | 0.7683 | 🏁 Pioneiros | Leve queda na acurácia, comportamento semelhante à regressão logística. |
| **Gradient Boosting** | 0.7645 | 🏁 Pioneiros | Desempenho consistente, porém com baixa sensibilidade às classes pequenas. |
| **SVC** | 0.7722 | 🏁 Pioneiros | Igual à Logistic Regression, indicando viés para classe majoritária. |

---

##  Análise das Métricas  

### 🔹 Desempenho Geral  
- **Acurácia Média:** ~77%  
- **Média Ponderada de f1-score:** 0.67  

Apesar da acurácia relativamente alta, todos os modelos **classificaram quase todos os pilotos como “Pioneiros”**, ignorando as demais classes.  

### 🔸 Macro vs Weighted Average  
- **Macro Average (0.22)** → mostra o fraco desempenho nas classes menores  
- **Weighted Average (0.67)** → inflada pela grande quantidade de exemplos da classe majoritária  

---

##  Interpretação dos Resultados  

1. **Desbalanceamento extremo** levou os modelos a **focar exclusivamente na classe dominante** (“Pioneiros”).  
2. **Classes raras** (“Clássicos”, “Modernos”, “Contemporâneos”) não tiveram exemplos suficientes para aprendizado robusto.  
3. Modelos como **Logistic Regression** e **SVC** apresentaram resultados idênticos, sugerindo **limitação pela separabilidade dos dados** e não pela complexidade do modelo.  
4. Mesmo algoritmos de *ensemble* (RandomForest, GradientBoosting) **não conseguiram generalizar** melhor sem reamostragem.  

---

##  Conclusão  

1. **Melhores Desempenhos:** Logistic Regression e SVC (Acurácia: 77,22%)  
2. **Desempenho Geral Limitado:** todos os modelos falharam em distinguir as classes menores  
3. **Principal Causa:** desbalanceamento da base de dados e baixa variabilidade das features para separar temporalmente os pilotos  

> Em resumo, os modelos foram capazes de identificar com precisão os **pilotos mais antigos**, mas **não conseguiram reconhecer as gerações seguintes**, indicando necessidade de ajustes nos dados e técnicas de balanceamento.  

---

## 🔧 Recomendações  

1. **Balanceamento de Classes:**  
   - Aplicar técnicas como **SMOTE**, **Random Oversampling** ou **Class Weights** para equilibrar as eras  

2. **Engenharia de Features:**  
   - Adicionar variáveis representativas como número de corridas, estreias, vitórias, equipes e décadas de atividade  

3. **Avaliação Alternativa:**  
   - Usar métricas como **F1-macro** e **Matriz de Confusão Normalizada** para evitar distorção por classes desbalanceadas  

4. **Modelos Alternativos:**  
   - Testar **XGBoost** e **CatBoost**, que lidam melhor com classes desbalanceadas  

5. **Pipeline Futuro:**  
   - Criar pipeline automatizado de *preprocessing → training → evaluation* com ajuste de hiperparâmetros via **GridSearchCV**  

---

##  Considerações Finais  
O estudo mostrou que **a acurácia sozinha não é suficiente** para avaliar modelos com classes desbalanceadas.  
Apesar do bom desempenho numérico, a análise detalhada revela **falta de aprendizado real nas classes menores**.  

Com ajustes de balanceamento e engenharia de variáveis, o modelo tem potencial de alcançar **classificações mais justas e representativas** das diferentes eras da Fórmula 1.  
