import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Exploração dos Dados
print("\n=== 1. EXPLORAÇÃO DOS DADOS ===\n")
df = pd.read_csv('docs/knn/drivers.csv')
print("Dimensão do dataset:", df.shape)
print("\nTipos de variáveis:\n", df.dtypes)
print("\nPrimeiras linhas:\n", df.head())
print("\nEstatísticas descritivas:\n", df.describe(include='all'))
print("\nValores nulos por coluna:\n", df.isnull().sum())

# Visualização de variáveis numéricas
df.select_dtypes(include=[np.number]).hist(bins=20, figsize=(12,8))
plt.suptitle("Distribuição das Variáveis Numéricas")
plt.show()

# Visualização de variáveis categóricas
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x=col, palette="viridis")
    plt.title(f"Distribuição da variável {col}")
    plt.xticks(rotation=45)
    plt.show()

# 2. Pré-processamento
print("\n=== 2. PRÉ-PROCESSAMENTO ===\n")
# Preencher valores ausentes
def fillna_col(col):
    if df[col].dtype in ["int64", "float64"]:
        return df[col].fillna(df[col].mean())
    else:
        return df[col].fillna(df[col].mode()[0])
for col in df.columns:
    df[col] = fillna_col(col)
print("Valores nulos após tratamento:\n", df.isnull().sum())

# Normalização das variáveis numéricas
num_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Codificação da variável target (exemplo: nationality)
le = LabelEncoder()
df['nationality_encoded'] = le.fit_transform(df['nationality'])

# 3. Divisão dos Dados
print("\n=== 3. DIVISÃO DOS DADOS ===\n")
X = df.drop(['nationality', 'nationality_encoded'], axis=1)
y = df['nationality_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Tamanho do conjunto de treino: {X_train.shape[0]}")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

# 4. Treinamento do Modelo
print("\n=== 4. TREINAMENTO DO MODELO KNN ===\n")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("Modelo KNN treinado com sucesso!")

# 5. Avaliação do Modelo
print("\n=== 5. AVALIAÇÃO DO MODELO ===\n")
y_pred = knn.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=le.inverse_transform(np.unique(y_test))))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.inverse_transform(np.unique(y_test)), yticklabels=le.inverse_transform(np.unique(y_test)))
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão - KNN')
plt.show()
