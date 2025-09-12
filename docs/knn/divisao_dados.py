
import pandas as pd
from sklearn.model_selection import train_test_split
from PréProcssamento import X_scaled, y, X

print("\n=== DIVISÃO DOS DADOS ===\n")

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Tamanho do conjunto original: {X.shape[0]} amostras")
print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

print("\nDistribuição das classes no conjunto de treino:")
print(pd.Series(y_train).value_counts())

print("\nDistribuição das classes no conjunto de teste:")
print(pd.Series(y_test).value_counts())