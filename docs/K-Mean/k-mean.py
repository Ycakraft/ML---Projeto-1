import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Caminhos
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "KNN", "drivers.csv")
output_svg = os.path.join(script_dir, "drivers_clusters.svg")
output_csv = os.path.join(script_dir, "drivers_clusters_result.csv")

print(f"Carregando dados de: {csv_path}")

try:
    df = pd.read_csv(csv_path)
    print(f"Colunas disponíveis no dataset: {df.columns.tolist()}")
    print(f"Primeiras linhas do dataset:\n{df.head()}")
    print(f"Número total de pilotos: {len(df)}")
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em {csv_path}")
    exit()
except Exception as e:
    print(f"Erro ao carregar arquivo: {e}")
    exit()

# Pré-processamento dos dados
print("\nPreparando dados para clustering...")

# Criar cópia para análise
drivers_df = df.copy()

# Converter data de nascimento para idade (aproximada)
current_year = datetime.now().year
drivers_df['dob'] = pd.to_datetime(drivers_df['dob'], errors='coerce')
drivers_df['age'] = current_year - drivers_df['dob'].dt.year

# Lidar com valores missing
drivers_df['age'] = drivers_df['age'].fillna(drivers_df['age'].median())

# Criar features para análise
# 1. Comprimento do nome (como proxy para experiência/geração)
drivers_df['name_length'] = (drivers_df['forename'] + ' ' + drivers_df['surname']).str.len()

# 2. Nacionalidade (encoded)
le_nationality = LabelEncoder()
drivers_df['nationality_encoded'] = le_nationality.fit_transform(drivers_df['nationality'])

# 3. Década de nascimento
drivers_df['birth_decade'] = (drivers_df['dob'].dt.year // 10) * 10

# 4. Tem piloto tem número permanente?
drivers_df['has_permanent_number'] = drivers_df['number'].notna()

# 5. Tem código de 3 letras?
drivers_df['has_code'] = drivers_df['code'].notna()

# Preparar matriz de features para clustering
features = [
    'age', 
    'name_length', 
    'nationality_encoded', 
    'birth_decade'
]

# Adicionar features binárias
drivers_df['permanent_number_flag'] = drivers_df['has_permanent_number'].astype(int)
drivers_df['code_flag'] = drivers_df['has_code'].astype(int)

features.extend(['permanent_number_flag', 'code_flag'])

print(f"Features utilizadas: {features}")

# Remover linhas com valores nulos nas features
X = drivers_df[features].dropna()

if len(X) == 0:
    print("Erro: Nenhum dado disponível após remoção de valores nulos.")
    exit()

print(f"Número de pilotos para clustering: {len(X)}")

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar número ótimo de clusters usando método do cotovelo
print("\nCalculando número ótimo de clusters...")
inertia = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot do método do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, 'bo-', alpha=0.7)
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para Determinação de K')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(script_dir, "elbow_method.svg"), format='svg', bbox_inches='tight')
plt.show()

# Escolher K baseado no gráfico (você pode ajustar manualmente)
K = 4
print(f"Usando K={K} clusters")

# Aplicar KMeans
print("Aplicando KMeans...")
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Adicionar clusters ao dataframe
drivers_df = drivers_df.loc[X.index].copy()
drivers_df['cluster'] = labels

# Análise dos clusters
print("\nAnalisando clusters...")

# Plot 2D dos clusters (usando PCA para visualização)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                      s=50, alpha=0.7, edgecolor='k', linewidth=0.5)

plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title(f'Clusters de Pilotos - KMeans (K={K})')
plt.colorbar(scatter, label='Cluster')

# Adicionar alguns rótulos de pilotos famosos
famous_drivers = ['Hamilton', 'Verstappen', 'Schumacher', 'Senna', 'Fangio']
for driver in famous_drivers:
    mask = drivers_df['surname'].str.contains(driver, case=False, na=False)
    if mask.any():
        idx = mask[mask].index[0]
        if idx in X.index:
            pca_idx = list(X.index).index(idx)
            plt.annotate(driver, (X_pca[pca_idx, 0], X_pca[pca_idx, 1]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.8, fontweight='bold')

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_svg, format='svg', bbox_inches='tight', dpi=300)
print(f"Gráfico salvo como: {output_svg}")
plt.show()

# Análise detalhada por cluster
print("\n" + "="*60)
print("ANÁLISE DETALHADA DOS CLUSTERS")
print("="*60)

# Estatísticas por cluster
cluster_stats = drivers_df.groupby('cluster').agg({
    'driverId': 'count',
    'age': ['mean', 'std', 'min', 'max'],
    'birth_decade': 'mean',
    'nationality': lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A',
    'has_permanent_number': 'mean',
    'has_code': 'mean'
}).round(2)

# Renomear colunas
cluster_stats.columns = [
    'count', 'age_mean', 'age_std', 'age_min', 'age_max',
    'birth_decade_mean', 'most_common_nationality',
    'pct_permanent_number', 'pct_has_code'
]

print(cluster_stats)

# Exemplos por cluster
print("\nEXEMPLOS DE PILOTOS POR CLUSTER:")
for cluster in sorted(drivers_df['cluster'].unique()):
    cluster_data = drivers_df[drivers_df['cluster'] == cluster]
    print(f"\n{'='*40}")
    print(f"CLUSTER {cluster} ({len(cluster_data)} pilotos)")
    print(f"{'='*40}")
    
    # Características do cluster
    avg_age = cluster_data['age'].mean()
    common_nationality = cluster_data['nationality'].mode().iloc[0]
    pct_modern = cluster_data['has_permanent_number'].mean() * 100
    
    print(f"Idade média: {avg_age:.1f} anos")
    print(f"Nacionalidade mais comum: {common_nationality}")
    print(f"Pilotos com número permanente: {pct_modern:.1f}%")
    
    # Exemplos de pilotos
    sample_drivers = cluster_data[['forename', 'surname', 'nationality', 'dob']].head(5)
    print("\nExemplos de pilotos:")
    for _, driver in sample_drivers.iterrows():
        birth_year = driver['dob'].year if pd.notna(driver['dob']) else 'N/A'
        print(f"  {driver['forename']} {driver['surname']} ({driver['nationality']}, {birth_year})")

# Heatmap de características por cluster
plt.figure(figsize=(12, 8))

# Preparar dados para heatmap
heatmap_data = drivers_df.groupby('cluster').agg({
    'age': 'mean',
    'birth_decade': 'mean', 
    'permanent_number_flag': 'mean',
    'code_flag': 'mean'
})

# Normalizar para heatmap
heatmap_data_normalized = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()

plt.subplot(1, 2, 1)
sns.heatmap(heatmap_data_normalized, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', cbar_kws={'label': 'Desvio Padrão'})
plt.title('Características dos Clusters (Normalizado)')

# Distribuição de nacionalidades por cluster
plt.subplot(1, 2, 2)
nationality_cluster = pd.crosstab(drivers_df['cluster'], 
                                 drivers_df['nationality'], 
                                 normalize='index')
# Pegar apenas as top 5 nacionalidades por cluster
top_nationalities = nationality_cluster.sum().nlargest(5).index
nationality_cluster_top = nationality_cluster[top_nationalities]

sns.heatmap(nationality_cluster_top, annot=True, cmap='YlOrRd', 
            fmt='.2f', cbar_kws={'label': 'Proporção'})
plt.title('Top Nacionalidades por Cluster')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "clusters_analysis.svg"), format='svg', bbox_inches='tight')
plt.show()

# Salvar resultados
output_df = drivers_df[['driverId', 'driverRef', 'forename', 'surname', 
                       'nationality', 'dob', 'age', 'cluster']]
output_df.to_csv(output_csv, index=False)
print(f"\nResultados salvos em: {output_csv}")

# Resumo final
print("\n" + "="*60)
print("RESUMO FINAL")
print("="*60)
print(f"Total de pilotos analisados: {len(drivers_df)}")
print(f"Número de clusters: {K}")

cluster_summary = drivers_df.groupby('cluster').size()
for cluster, count in cluster_summary.items():
    percentage = (count / len(drivers_df)) * 100
    print(f"Cluster {cluster}: {count} pilotos ({percentage:.1f}%)")

print(f"\nAnálise concluída! Verifique os arquivos gerados:")
print(f"- {output_svg} (Gráfico principal)")
print(f"- {output_csv} (Dados com clusters)")
print(f"- {os.path.join(script_dir, 'clusters_analysis.svg')} (Análise detalhada)")
print(f"- {os.path.join(script_dir, 'elbow_method.svg')} (Método do cotovelo)")