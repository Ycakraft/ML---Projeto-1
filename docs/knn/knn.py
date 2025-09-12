import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Carregar dados do drivers.csv
data = pd.read_csv('drivers.csv')

# Verificar as colunas disponíveis
print("Colunas disponíveis no dataset:")
print(data.columns.tolist())
print(f"\nShape do dataset: {data.shape}")
print("\nPrimeiras 5 linhas:")
print(data.head())

# Encontrar colunas numéricas para features
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nColunas numéricas: {numeric_columns}")

# Encontrar colunas categóricas para target
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
print(f"Colunas categóricas: {categorical_columns}")

# Selecionar as duas primeiras colunas numéricas para features
if len(numeric_columns) >= 2:
    feature1, feature2 = numeric_columns[0], numeric_columns[1]
    print(f"\nUsando features: {feature1} e {feature2}")
else:
    raise ValueError("Não há colunas numéricas suficientes no dataset")

# Selecionar a primeira coluna categórica para target
if len(categorical_columns) > 0:
    target_column = categorical_columns[0]
    print(f"Usando como target: {target_column}")
else:
    # Se não houver colunas categóricas, criar uma baseada em quartis
    print("Criando target artificial baseado na primeira coluna numérica...")
    target_column = 'category'
    data[target_column] = pd.qcut(data[numeric_columns[0]], q=3, labels=['Low', 'Medium', 'High'])

# Preparar os dados
X = data[[feature1, feature2]].values
y = data[target_column]

# Converter target para numérico se for categórico
if y.dtype == 'object':
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
else:
    y_encoded = y.values
    class_names = [f'Class {i}' for i in np.unique(y_encoded)]

# Remover NaN values se houver
mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_encoded)
X = X[mask]
y_encoded = y_encoded[mask]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar o classificador KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y_encoded)

# Configurar o mesh grid para plotar o limite de decisão
h = 0.02  # tamanho do passo no mesh
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Prever classes para cada ponto no mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Configurar o plot
plt.figure(figsize=(12, 9))

# Criar mapa de cores baseado no número de classes
n_classes = len(np.unique(y_encoded))
colors_light = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFAAFF', '#FFFFAA', '#AAFFFF']
colors_bold = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF']

cmap_light = ListedColormap(colors_light[:n_classes])
cmap_bold = ListedColormap(colors_bold[:n_classes])

# Plotar o limite de decisão
plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)

# Plotar os pontos de treinamento
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_encoded, cmap=cmap_bold, 
                      edgecolor='black', s=60, alpha=0.8)

# Configurar título e labels
plt.title(f"KNN Decision Boundary (k=3)\nFeatures: {feature1} vs {feature2}\nTarget: {target_column}", 
          fontsize=14, fontweight='bold')
plt.xlabel(f"{feature1} (normalized)", fontsize=12)
plt.ylabel(f"{feature2} (normalized)", fontsize=12)

# Adicionar grid
plt.grid(True, linestyle='--', alpha=0.6)

# Adicionar legenda
handles, _ = scatter.legend_elements()
plt.legend(handles, class_names, title=target_column, 
           loc="best", fontsize=10)

# Adicionar informações no gráfico
info_text = f'Total samples: {len(X)}\nNumber of classes: {n_classes}'
plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Ajustar layout
plt.tight_layout()

# Mostrar o plot
plt.show()

# Mostrar estatísticas
print(f"\n=== ESTATÍSTICAS ===")
print(f"Dataset shape: {data.shape}")
print(f"Features usadas: {feature1}, {feature2}")
print(f"Target: {target_column}")
print(f"Número de classes: {n_classes}")
print(f"Distribuição das classes:")
for i, class_name in enumerate(class_names):
    count = np.sum(y_encoded == i)
    print(f"  {class_name}: {count} samples ({count/len(y_encoded)*100:.1f}%)")