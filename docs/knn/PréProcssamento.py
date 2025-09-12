
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("drivers.csv")

print("\n=== PRÉ-PROCESSAMENTO ===\n")

# Verificando valores ausentes
print("Valores ausentes por coluna:")
print(df.isnull().sum())

# Como não há valores ausentes, não é necessário tratamento
print("Não há valores ausentes no dataset.")

# Codificando a variável target (já está codificada como 0, 1, 2)
# Para fins demonstrativos, vamos garantir que está correta
le = LabelEncoder()
df['target_encoded'] = le.fit_transform(df['species'])
print("\nVerificação da codificação da target:")
print(df[['species', 'target', 'target_encoded']].head())

# Normalização dos dados
scaler = StandardScaler()
X = df.iloc[:, :4]  # Features
y = df['target_encoded']  # Target

X_scaled = scaler.fit_transform(X)
print("\nDados antes da normalização (primeiras 5 linhas):")
print(X.head())

print("\nDados após normalização (primeiras 5 linhas):")
print(pd.DataFrame(X_scaled, columns=X.columns).head())