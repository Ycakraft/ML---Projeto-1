import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ============================
# 1. Carregar dataset (exemplo)
# ============================
df = pd.read_csv("drivers.csv")

# ============================
# 2. Remover colunas duplicadas/irrelevantes
# ============================
df = df.loc[:, ~df.columns.duplicated()]

# ============================
# 3. Tratar valores ausentes
# ============================
# Exemplo: preencher valores numéricos com média e categóricos com moda
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# ============================
# 4. Normalização / Padronização
# ============================
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Normalização Min-Max (0 a 1)
scaler_minmax = MinMaxScaler()
df[num_cols] = scaler_minmax.fit_transform(df[num_cols])

# OU: Padronização Z-Score (média=0, desvio=1)
# scaler_standard = StandardScaler()
# df[num_cols] = scaler_standard.fit_transform(df[num_cols])

# ============================
# 5. Conferir dataset pronto
# ============================
print("📌 Dataset após pré-processamento:")
print(df.head())
print("\n📌 Valores nulos após tratamento:\n", df.isnull().sum())
print("\n📌 Estatísticas descritivas:\n", df.describe(include="all"))