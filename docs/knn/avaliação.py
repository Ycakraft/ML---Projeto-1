import pandas as pd
import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Usar arquivo local já baixado
csv_path = "../../../../formula-1-world-championship-1950-2020/drivers.csv"

# Carregar dados
df = pd.read_csv(csv_path)

# Tratamento de valores ausentes
for col in df.columns:
    if df[col].dtype in ["int64","float64"]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Normalização das colunas numéricas
num_cols = df.select_dtypes(include=["int64","float64"]).columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Filtrar nacionalidades com pelo menos 2 ocorrências
counts = df['nationality'].value_counts()
valid_nationalities = counts[counts >= 2].index
df_filtered = df[df['nationality'].isin(valid_nationalities)]

# Preparar dados para o modelo
target_column = "nationality"

# Codificar variáveis categóricas (exceto a target)
categorical_cols = df_filtered.select_dtypes(include=['object']).columns
categorical_cols = categorical_cols.drop(target_column, errors='ignore')

X = df_filtered.drop(target_column, axis=1)
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
y = df_filtered[target_column]

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Criar e treinar modelo KNN
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo KNN: {accuracy:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.title('Matriz de Confusão - KNN')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# Encontrar o melhor valor de k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    accuracy_temp = accuracy_score(y_test, y_pred_temp)
    accuracies.append(accuracy_temp)

# Plotar acurácia vs valor de k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.title('Acurácia do KNN para diferentes valores de k')
plt.grid(True)
plt.show()

# Melhor valor de k
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"Melhor valor de k: {best_k}")
print(f"Melhor acurácia: {best_accuracy:.2f}")

# Treinar modelo com melhor k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred_best = best_knn.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_best)

print(f"\nAcurácia final com k={best_k}: {final_accuracy:.2f}")