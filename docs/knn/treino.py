
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from docs.knn.divisao_dados import X_train, X_test, y_train, y_test # pyright: ignore[reportMissingImports]

print("\n=== TREINAMENTO DO MODELO KNN ===\n")

# Encontrando o melhor valor de k
accuracies = []
k_values = range(1, 21)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Plotando a acurácia para diferentes valores de k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.title('Acurácia do Modelo KNN para Diferentes Valores de k')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Selecionando o melhor k
best_k = k_values[np.argmax(accuracies)]
print(f"Melhor valor de k: {best_k} com acurácia de {max(accuracies):.4f}")

# Treinando o modelo com o melhor k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

print("Modelo KNN treinado com sucesso!")