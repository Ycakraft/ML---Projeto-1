import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_iris

# Carregando conjunto de dados de exemplo (Iris dataset)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['species'] = df['target'].apply(lambda x: data.target_names[x])

print("=== EXPLORAÇÃO DOS DADOS ===\n")
print("Natureza dos dados:")
print("O conjunto de dados Iris é clássico na área de machine learning.")
print("Contém medidas de sépalas e pétalas de três espécies de flores Iris:")
print("- Setosa (0), Versicolor (1) e Virginica (2)\n")

print("Dimensões do dataset:", df.shape)
print("\nPrimeiras 5 linhas:")
print(df.head())

print("\nInformações do dataset:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

print("\nDistribuição das classes:")
print(df['species'].value_counts())

# Visualizações
plt.figure(figsize=(15, 10))

# Histogramas
plt.subplot(2, 3, 1)
df['sepal length (cm)'].hist()
plt.title('Distribuição do Comprimento da Sépala')

plt.subplot(2, 3, 2)
df['sepal width (cm)'].hist()
plt.title('Distribuição da Largura da Sépala')

plt.subplot(2, 3, 3)
df['petal length (cm)'].hist()
plt.title('Distribuição do Comprimento da Pétala')

plt.subplot(2, 3, 4)
df['petal width (cm)'].hist()
plt.title('Distribuição da Largura da Pétala')

plt.subplot(2, 3, 5)
df['species'].value_counts().plot(kind='bar')
plt.title('Distribuição das Espécies')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Matriz de correlação
plt.figure(figsize=(8, 6))
correlation_matrix = df.iloc[:, :4].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação entre as Variáveis')
plt.show()

# Scatter plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species')
plt.title('Sépala: Comprimento vs Largura')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species')
plt.title('Pétala: Comprimento vs Largura')

plt.tight_layout()
plt.show()