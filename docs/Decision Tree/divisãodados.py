# ============================
# Pré-requisitos
# ============================
# pip install pandas matplotlib seaborn scikit-learn opendatasets

import opendatasets as od
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ============================
# 1. Escolha do CSV público do Kaggle
# ============================
# Exemplo: dataset de Fórmula 1
dataset_url = "https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020?resource=download"
od.download(dataset_url)

# Caminho do CSV que você quer usar
csv_path = "formula-1-world-championship-1950-2020/drivers.csv"  # ajuste conforme o dataset

# ============================
# 2. Carregar CSV
# ============================
df = pd.read_csv(csv_path)

# ============================
# 3. EDA básica
# ============================
print("📌 Dimensão do dataset:", df.shape)
print("\n📌 Tipos de variáveis:\n", df.dtypes)
print("\n📌 Valores nulos por coluna:\n", df.isnull().sum())
print("\n📌 Estatísticas gerais:\n", df.describe(include="all"))

# Histograma das variáveis numéricas
df.hist(bins=20, figsize=(12,8))
plt.suptitle("Distribuição das Variáveis Numéricas")
plt.show()

# Heatmap de correlação numérica
num_cols = df.select_dtypes(include=["int64","float64"]).columns
if len(num_cols) > 1:
    plt.figure(figsize=(10,6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlação")
    plt.show()

# Distribuição de variáveis categóricas
for col in df.select_dtypes(include="object").columns:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x=col, palette="viridis")
    plt.title(f"Distribuição da variável {col}")
    plt.xticks(rotation=45)
    plt.show()

# ============================
# 4. Pré-processamento
# ============================
df = df.loc[:, ~df.columns.duplicated()]

for col in df.columns:
    if df[col].dtype in ["int64","float64"]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\n📌 Dataset após pré-processamento:")
print(df.head())
print("\n📌 Valores nulos após tratamento:\n", df.isnull().sum())

# ============================
# 5. Divisão treino/teste
# ============================
target_column = "nationality" 
X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📌 Shapes após divisão:")
print("Treino X:", X_train.shape)
print("Teste X:", X_test.shape)
print("Treino y:", y_train.shape)
print("Teste y:", y_test.shape)
