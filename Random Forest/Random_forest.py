import os
import pandas as pd
import numpy as np
from datetime import datetime

# Imports do Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Modelos para comparar
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Imports de Visualização (Interativa)
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Configura o tema padrão do Plotly para um visual limpo
pio.templates.default = "plotly_white"

# =============================
# CONFIGURAÇÕES GLOBAIS
# =============================
# Tenta obter o diretório do script, senão usa o diretório atual (para notebooks)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    print("⚠️  Executando em ambiente interativo. Usando o diretório atual (os.getcwd()).")
    script_dir = os.getcwd()

CSV_PATH = os.path.join(script_dir, "drivers.csv")
OUTPUT_DIR = os.path.join(script_dir, "model_comparison_report")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CLASS_NAMES = ['🏁 Pioneiros (<1960)', '⚡ Clássicos (60-70s)', '🚀 Modernos (80-90s)', '🎯 Contemporâneos (2000+)']


# =============================
# FUNÇÃO 1: CARREGAR DADOS
# =============================
def load_data(path):
    """Carrega o CSV com tratamento de erro."""
    print(f"🔄 Carregando dados de {path}...")
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset carregado: {len(df)} linhas, {len(df.columns)} colunas.")
        return df
    except Exception as e:
        print(f"❌ ERRO FATAL: Não foi possível carregar o arquivo: {e}")
        print(f"ℹ️  Verifique se o arquivo 'drivers.csv' está na pasta: {script_dir}")
        exit()

# =============================
# FUNÇÃO 2: PRÉ-PROCESSAMENTO E FEATURE ENGINEERING
# =============================
def preprocess_data(df):
    """Aplica feature engineering e separa X (features) e y (target)."""
    print("🛠️  Iniciando pré-processamento e feature engineering...")
    
    # Copia para evitar SettingWithCopyWarning
    data = df.copy()
    
    # --- CRIAÇÃO DO TARGET (y) ---
    data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
    data['birth_year'] = data['dob'].dt.year
    # Preenche NaT (Not a Time) com a mediana para não perder dados
    data['birth_year'] = data['birth_year'].fillna(data['birth_year'].median())
    
    # Define a 'era' (nosso target)
    data['era'] = data['birth_year'].apply(
        lambda x: 0 if x < 1960 else (1 if x < 1980 else (2 if x < 2000 else 3))
    )
    
    # --- CRIAÇÃO DAS FEATURES (X) ---
    # *NOTA CRÍTICA: Não usaremos 'age' ou 'birth_year' como feature
    # para evitar data leakage (prever a era usando a própria data de nasc.)
    
    # 1. Tamanho do nome
    data['name_length'] = (data['forename'] + data['surname']).str.len()
    
    # 2. Features booleanas
    data['has_number'] = data['number'].notna().astype(int)
    data['has_code'] = data['code'].notna().astype(int)
    
    # 3. Nacionalidade (agrupada)
    nationality_counts = data['nationality'].value_counts()
    common_nationalities = nationality_counts[nationality_counts > 5].index
    data['nationality_group'] = data['nationality'].apply(
        lambda x: x if x in common_nationalities else 'Other'
    )
    
    # Codifica a nacionalidade
    le = LabelEncoder()
    data['nationality_encoded'] = le.fit_transform(data['nationality_group'])
    
    # --- DEFINIÇÃO FINAL ---
    feature_cols = ['name_length', 'has_number', 'has_code', 'nationality_encoded']
    target_col = 'era'
    
    X = data[feature_cols]
    y = data[target_col]
    
    print(f"📊 Features selecionadas (X): {feature_cols}")
    print(f"🎯 Target selecionado (y): {target_col}")
    
    return X, y, feature_cols

# =============================
# FUNÇÃO 3: TREINAMENTO E TUNING DE MODELOS
# =============================
def train_models(X_train, y_train):
    """Cria pipelines e usa GridSearchCV para treinar múltiplos modelos."""
    print("\n🤖 Iniciando treinamento e tuning de múltiplos modelos...")
    
    # Define os modelos e os hiperparâmetros para testar
    models_config = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, multi_class='ovr'),
            'params': {
                'model__C': [0.1, 1.0, 10.0],
                'model__solver': ['liblinear']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'model__n_estimators': [50, 100, 150],
                'model__max_depth': [3, 5, 10],
                'model__min_samples_leaf': [2, 4]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5]
            }
        },
        'SVC': {
            'model': SVC(random_state=RANDOM_STATE, probability=True),
            'params': {
                'model__C': [0.1, 1.0, 10.0],
                'model__kernel': ['linear', 'rbf']
            }
        }
    }
    
    trained_models = {}
    
    for model_name, config in models_config.items():
        print(f"--- Treinando {model_name} ---")
        
        # Cria um pipeline que primeiro padroniza os dados, depois aplica o modelo
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', config['model'])
        ])
        
        # Usa GridSearchCV para encontrar os melhores parâmetros
        grid_search = GridSearchCV(
            pipeline,
            param_grid=config['params'],
            cv=5,  # 5-fold cross-validation
            scoring='accuracy',
            n_jobs=-1, # Usa todos os processadores
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"🏆 Melhor Acurácia (CV): {grid_search.best_score_:.4f}")
        print(f"🔩 Melhores Parâmetros: {grid_search.best_params_}")
        
        trained_models[model_name] = grid_search
        
    print("✅ Treinamento concluído.")
    return trained_models

# =============================
# FUNÇÃO 4: AVALIAÇÃO E GERAÇÃO DE RELATÓRIO
# =============================
def generate_report(trained_models, X_test, y_test, feature_names):
    """Avalia os modelos no set de teste e gera visualizações interativas."""
    print("\n📈 Gerando relatório de performance...")
    
    results = []
    
    # --- 1. Coleta de Resultados e Identificação do Melhor Modelo ---
    best_accuracy = -1
    best_model_name = ""
    best_model_pipeline = None

    print("\n" + "="*30 + " RELATÓRIO DE TESTE " + "="*30)
    for model_name, model_pipeline in trained_models.items():
        y_pred = model_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({'Modelo': model_name, 'Acurácia no Teste': accuracy})
        
        print(f"\n--- {model_name} (Acurácia: {accuracy:.4f}) ---")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_model_pipeline = model_pipeline
            
    print("="*82)
    print(f"🥇 Melhor Modelo Geral: {best_model_name} (Acurácia: {best_accuracy:.4f})")
    
    results_df = pd.DataFrame(results).sort_values(by='Acurácia no Teste', ascending=False)
    
    # --- 2. Visualização: Comparação de Modelos ---
    fig_comparison = px.bar(
        results_df,
        x='Modelo',
        y='Acurácia no Teste',
        color='Modelo',
        title='Comparação de Performance dos Modelos (Dados de Teste)',
        text_auto='.4f'
    )
    fig_comparison.update_layout(showlegend=False)
    fig_comparison.write_html(os.path.join(OUTPUT_DIR, "1_model_comparison.html"))
    print(f"💾 Gráfico '1_model_comparison.html' salvo.")

    # --- 3. Visualização: Matriz de Confusão do Melhor Modelo ---
    y_pred_best = best_model_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Previsto", y="Verdadeiro", color="Contagem"),
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        text_auto=True,
        color_continuous_scale='Blues',
        title=f"Matriz de Confusão - {best_model_name}"
    )
    fig_cm.write_html(os.path.join(OUTPUT_DIR, "2_confusion_matrix.html"))
    print(f"💾 Gráfico '2_confusion_matrix.html' salvo.")

    # --- 4. Visualização: Importância das Features do Melhor Modelo ---
    best_model_object = best_model_pipeline.best_estimator_.named_steps['model']
    
    importances = None
    if hasattr(best_model_object, 'feature_importances_'):
        # Para RandomForest e GradientBoosting
        importances = best_model_object.feature_importances_
    elif hasattr(best_model_object, 'coef_'):
        # Para LogisticRegression e SVC(kernel='linear')
        # Média do valor absoluto dos coeficientes (para casos multi-classe)
        if best_model_object.coef_.ndim > 1:
             importances = np.mean(np.abs(best_model_object.coef_), axis=0)
        else:
             importances = np.abs(best_model_object.coef_)

    if importances is not None:
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importância': importances})
        importance_df = importance_df.sort_values(by='Importância', ascending=True)
        
        fig_importance = px.bar(
            importance_df,
            x='Importância',
            y='Feature',
            orientation='h',
            title=f"Importância das Features - {best_model_name}"
        )
        fig_importance.write_html(os.path.join(OUTPUT_DIR, "3_feature_importance.html"))
        print(f"💾 Gráfico '3_feature_importance.html' salvo.")
    else:
        print(f"ℹ️  Não foi possível extrair importância das features para o modelo {best_model_name} (ex: SVC com kernel RBF).")

# =============================
# FUNÇÃO PRINCIPAL (MAIN)
# =============================
def main():
    """Orquestra todo o fluxo de trabalho de ML."""
    
    print("🚀 Iniciando Pipeline de Comparação de Modelos...")
    print(f"📂 Os relatórios serão salvos em: {OUTPUT_DIR}")
    
    # 1. Carga
    raw_df = load_data(CSV_PATH)
    
    # 2. Processamento
    X, y, feature_names = preprocess_data(raw_df)
    
    if len(X) == 0:
        print("❌ ERRO: Nenhum dado restou após o processamento.")
        return

    # 3. Divisão
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n📏 Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    # 4. Treinamento
    trained_models = train_models(X_train, y_train)
    
    # 5. Relatório
    generate_report(trained_models, X_test, y_test, feature_names)
    
    print("\n🎉 Análise concluída com sucesso!")
    print(f"👉 Abra os arquivos .html em '{OUTPUT_DIR}' para ver os resultados interativos.")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    main()