import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

class KNNClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN Classifier - Drivers Dataset")
        self.root.geometry("1200x800")
        
        # Variáveis
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.create_widgets()
        self.load_data()
    
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        title_label = ttk.Label(main_frame, text="KNN Classifier - Drivers Dataset", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Botões de controle
        self.load_btn = ttk.Button(main_frame, text="Carregar Dados", command=self.load_data)
        self.load_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.train_btn = ttk.Button(main_frame, text="Treinar Modelo", command=self.train_model)
        self.train_btn.grid(row=1, column=1, padx=5, pady=5)
        
        self.predict_btn = ttk.Button(main_frame, text="Fazer Previsão", command=self.predict_dialog)
        self.predict_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # Parâmetros do KNN
        ttk.Label(main_frame, text="Número de Vizinhos (k):").grid(row=2, column=0, padx=5, pady=5)
        self.k_var = tk.StringVar(value="5")
        self.k_entry = ttk.Entry(main_frame, textvariable=self.k_var, width=10)
        self.k_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Informações do dataset
        self.info_text = tk.Text(main_frame, height=8, width=80)
        self.info_text.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Frame para gráficos
        graph_frame = ttk.Frame(main_frame)
        graph_frame.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Figura para os gráficos
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Resultados
        self.result_text = tk.Text(main_frame, height=10, width=80)
        self.result_text.grid(row=5, column=0, columnspan=4, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Configurar expansão
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.columnconfigure(3, weight=1)
        main_frame.rowconfigure(4, weight=1)
    
    def load_data(self):
        try:
            # Carregar dados do drivers.csv
            # Se o arquivo não existir, criar dados de exemplo
            try:
                self.data = pd.read_csv('drivers.csv')
            except FileNotFoundError:
                self.create_sample_data()
            
            # Exibir informações do dataset
            info = f"Dataset carregado com sucesso!\n"
            info += f"Shape: {self.data.shape}\n"
            info += f"Colunas: {list(self.data.columns)}\n"
            info += f"Primeiras 5 linhas:\n{self.data.head().to_string()}\n"
            info += f"\nInformações:\n{self.data.info()}\n"
            info += f"\nDescrição estatística:\n{self.data.describe().to_string()}"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info)
            
            # Plotar gráficos iniciais
            self.plot_initial_graphs()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados: {str(e)}")
    
    def create_sample_data(self):
        """Cria dados de exemplo se o arquivo drivers.csv não existir"""
        np.random.seed(42)
        n_samples = 1000
        
        # Criar dados de exemplo para motoristas
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'experience_years': np.random.randint(0, 50, n_samples),
            'annual_mileage': np.random.randint(5000, 50000, n_samples),
            'accidents_last_year': np.random.randint(0, 5, n_samples),
            'speeding_tickets': np.random.randint(0, 10, n_samples),
            'risk_level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.6, 0.3, 0.1])
        }
        
        self.data = pd.DataFrame(data)
        self.data.to_csv('drivers.csv', index=False)
        messagebox.showinfo("Info", "Arquivo drivers.csv criado com dados de exemplo!")
    
    def plot_initial_graphs(self):
        """Plotar gráficos iniciais do dataset"""
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Gráfico 1: Distribuição das idades
        self.axes[0].hist(self.data['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        self.axes[0].set_title('Distribuição das Idades')
        self.axes[0].set_xlabel('Idade')
        self.axes[0].set_ylabel('Frequência')
        
        # Gráfico 2: Distribuição dos níveis de risco
        risk_counts = self.data['risk_level'].value_counts()
        self.axes[1].bar(risk_counts.index, risk_counts.values, color=['green', 'orange', 'red'])
        self.axes[1].set_title('Distribuição dos Níveis de Risco')
        self.axes[1].set_xlabel('Nível de Risco')
        self.axes[1].set_ylabel('Contagem')
        
        self.canvas.draw()
    
    def train_model(self):
        try:
            if self.data is None:
                messagebox.showerror("Erro", "Por favor, carregue os dados primeiro!")
                return
            
            # Preparar dados
            X = self.data.drop('risk_level', axis=1)
            y = self.data['risk_level']
            
            # Codificar variável target se for categórica
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)
            
            # Dividir dados
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Normalizar dados
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            # Treinar modelo KNN
            k = int(self.k_var.get())
            self.model = KNeighborsClassifier(n_neighbors=k)
            self.model.fit(self.X_train, self.y_train)
            
            # Fazer previsões
            y_pred = self.model.predict(self.X_test)
            
            # Calcular acurácia
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Exibir resultados
            result = f"Modelo KNN treinado com sucesso!\n"
            result += f"Número de vizinhos (k): {k}\n"
            result += f"Acurácia do modelo: {accuracy:.4f}\n\n"
            result += "Relatório de Classificação:\n"
            result += classification_report(self.y_test, y_pred, 
                                          target_names=self.label_encoder.classes_ if hasattr(self.label_encoder, 'classes_') else ['Class 0', 'Class 1', 'Class 2'])
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result)
            
            # Plotar matriz de confusão
            self.plot_confusion_matrix(y_pred)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar modelo: {str(e)}")
    
    def plot_confusion_matrix(self, y_pred):
        """Plotar matriz de confusão"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Matriz de confusão
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.axes[0])
        self.axes[0].set_title('Matriz de Confusão')
        self.axes[0].set_xlabel('Predito')
        self.axes[0].set_ylabel('Real')
        
        # Gráfico de importância das features (para KNN, podemos usar a acurácia por feature)
        self.plot_feature_importance()
        
        self.canvas.draw()
    
    def plot_feature_importance(self):
        """Plotar importância das features (simulado para KNN)"""
        # Para KNN, não temos importância direta das features, então usamos uma aproximação
        feature_names = self.data.drop('risk_level', axis=1).columns
        
        # Calcular acurácia removendo cada feature individualmente
        accuracies = []
        for i in range(len(feature_names)):
            X_temp = np.delete(self.X_train, i, axis=1)
            X_test_temp = np.delete(self.X_test, i, axis=1)
            
            model_temp = KNeighborsClassifier(n_neighbors=int(self.k_var.get()))
            model_temp.fit(X_temp, self.y_train)
            y_pred_temp = model_temp.predict(X_test_temp)
            acc = accuracy_score(self.y_test, y_pred_temp)
            accuracies.append(acc)
        
        # Normalizar importâncias
        base_accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
        importances = [base_accuracy - acc for acc in accuracies]
        
        self.axes[1].barh(feature_names, importances)
        self.axes[1].set_title('Importância das Features (Quanto maior, mais importante)')
        self.axes[1].set_xlabel('Redução na Acurácia quando removida')
    
    def predict_dialog(self):
        """Abrir diálogo para fazer previsões"""
        if self.model is None:
            messagebox.showerror("Erro", "Por favor, treine o modelo primeiro!")
            return
        
        # Criar janela de previsão
        predict_window = tk.Toplevel(self.root)
        predict_window.title("Fazer Previsão")
        predict_window.geometry("400x300")
        
        # Campos de entrada
        features = self.data.drop('risk_level', axis=1).columns
        entries = {}
        
        for i, feature in enumerate(features):
            ttk.Label(predict_window, text=f"{feature}:").grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
            entry = ttk.Entry(predict_window)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entries[feature] = entry
        
        # Botão de previsão
        def make_prediction():
            try:
                # Coletar dados
                input_data = []
                for feature in features:
                    value = float(entries[feature].get())
                    input_data.append(value)
                
                # Fazer previsão
                input_array = np.array([input_data])
                input_scaled = self.scaler.transform(input_array)
                prediction = self.model.predict(input_scaled)
                
                # Decodificar se necessário
                if hasattr(self.label_encoder, 'classes_'):
                    prediction_label = self.label_encoder.inverse_transform(prediction)[0]
                else:
                    prediction_label = prediction[0]
                
                # Mostrar resultado
                result_label.config(text=f"Previsão: {prediction_label}")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro na previsão: {str(e)}")
        
        ttk.Button(predict_window, text="Prever", command=make_prediction).grid(
            row=len(features), column=0, columnspan=2, pady=10
        )
        
        # Label para resultado
        result_label = ttk.Label(predict_window, text="Previsão: ", font=("Arial", 12, "bold"))
        result_label.grid(row=len(features)+1, column=0, columnspan=2, pady=10)

def main():
    root = tk.Tk()
    app = KNNClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()