
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from docs.knn.divisao_dados import X_test, y_test, X_train, y_train # pyright: ignore[reportMissingImports]
from treino import knn

print("\n=== AVALIAÇÃO DO MODELO ===\n")


# Fazendo previsões
y_pred = knn.predict(X_test)

# Métricas de avaliação
print("Acurácia do modelo:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, 
            yticklabels=labels)
plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.show()

# Análise adicional - Previsões vs Valores reais
results = pd.DataFrame({
    'Real': y_test,
    'Predito': y_pred
})

print("\nPrimeiras 10 previsões vs valores reais:")
print(results.head(10))

# Calculando acurácia por classe
correct_predictions = results[results['Real'] == results['Predito']]
accuracy_by_class = correct_predictions['Real'].value_counts() / results['Real'].value_counts()
print("\nAcurácia por classe:")
for classe, acc in accuracy_by_class.items():
    print(f"{classe}: {acc:.4f}")