# ============================
# Pré-requisitos
# ============================
# pip install pandas scikit-learn matplotlib opendatasets

import opendatasets as od
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# ============================
# 1. Baixar dataset drivers.csv do Kaggle
# ============================
dataset_url = "https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020"
od.download(dataset_url)
csv_path = "formula-1-world-championship-1950-2020/drivers.csv"

# ============================
# 2. Carregar CSV
# ============================
df = pd.read_csv(csv_path)

# ============================
# 3. Pré-processamento
# ============================
# Tratar nulos
for col in df.columns:
    if df[col].dtype in ["int64","float64"]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Normalização das colunas numéricas
num_cols = df.select_dtypes(include=["int64","float64"]).columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Filtrar nacionalidades com pelo menos 2 pilotos
counts = df['nationality'].value_counts()
valid_nationalities = counts[counts >= 2].index
df_filtered = df[df['nationality'].isin(valid_nationalities)]

# Features e target
target_column = "nationality"
X = pd.get_dummies(df_filtered.drop(target_column, axis=1), drop_first=True)
y = df_filtered[target_column]

# ============================
# 4. Divisão treino/teste
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 5. Treinamento Decision Tree
# ============================
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# ============================
# 6. Avaliação do modelo
# ============================
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"📌 Acurácia do modelo: {accuracy:.2f}")

print("\n📌 Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\n📌 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Visualização da árvore
plt.figure(figsize=(20,10))
tree.plot_tree(
    clf,
    feature_names=X_train.columns,
    class_names=clf.classes_,
    filled=True,
    rounded=True
)
plt.show()
