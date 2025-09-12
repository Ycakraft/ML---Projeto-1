


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Carregar o dataset Formula 1 drivers
df = pd.read_csv('docs/knn/drivers.csv')

# Pré-processamento: tratar nulos e normalizar
def fillna_col(col):
	if df[col].dtype in ["int64", "float64"]:
		return df[col].fillna(df[col].mean())
	else:
		return df[col].fillna(df[col].mode()[0])
for col in df.columns:
	df[col] = fillna_col(col)

num_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Codificação da variável target (nationality)
le = LabelEncoder()
df['nationality_encoded'] = le.fit_transform(df['nationality'])

# Divisão dos dados
X = df.drop(['nationality', 'nationality_encoded'], axis=1)
y = df['nationality_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Treinamento do modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Avaliação
y_pred = knn.predict(X_test)
print('Acurácia:', accuracy_score(y_test, y_pred))
print('\nRelatório de Classificação:')
print(classification_report(y_test, y_pred, target_names=le.inverse_transform(np.unique(y_test))))
