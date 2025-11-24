Relatório Final – Engenharia de Dados e Machine Learning com PySpark

1. Introdução

O objetivo deste projeto foi estabelecer um pipeline de Big Data local utilizando o framework Apache Spark (PySpark). A tarefa consistiu em carregar dados brutos de pilotos de Fórmula 1, realizar engenharia de atributos (cálculo de idade) e aplicar um algoritmo de aprendizado de máquina não supervisionado (K-Means) para agrupar os pilotos em gerações distintas (Novos Talentos, Auge, Veteranos).

2. Tecnologias Utilizadas

Python 3.x – interface de programação principal.

PySpark (Spark SQL & MLlib) – processamento distribuído e biblioteca de machine learning.

Java (JDK 8/11) – ambiente de execução necessário para a JVM do Spark.

Matplotlib – visualização gráfica dos clusters.

Pandas – utilizado apenas na etapa final para facilitar a plotagem dos dados agregados.

3. Etapas do Processo

3.1 Configuração do Ambiente e Dados

O ambiente Spark foi inicializado em modo local (.master("local[*]")), permitindo o uso de todos os núcleos da CPU para simular um cluster distribuído.
Os dados foram gerados dinamicamente em um arquivo drivers.csv para simular a ingestão de dados brutos.

3.2 Carregamento e Limpeza (ETL)

Leitura: O arquivo CSV foi lido utilizando spark.read.csv com a opção inferSchema=True para detecção automática de tipos numéricos e datas.

Limpeza: Registros com datas de nascimento (dob) nulas foram removidos (dropna) para garantir a consistência do cálculo de idade.

3.3 Engenharia de Atributos (Feature Engineering)

Criou-se uma nova variável (feature) fundamental para a análise:

age (Idade): Calculada através da função datediff, subtraindo a data de nascimento da data atual (current_date) e dividindo por 365.25 para obter a idade em anos.

3.4 Preparação para ML (Vetorização)

Diferente do scikit-learn, o Spark ML exige que as features de entrada estejam em um formato vetorial único.

VectorAssembler: Transformou a coluna escalar age em um vetor denso na coluna features, preparando o dataset para o algoritmo.

3.5 Modelagem

Algoritmo: K-Means Clustering (KMeans).

Hiperparâmetros: k=3 (número de clusters) e seed=1 (para reprodutibilidade).

Treinamento: O modelo foi ajustado (fit) aos dados vetorizados para encontrar os centróides ideais.

3.6 Visualização e Resultados

Para visualizar os dados processados pelo Spark:

Os resultados (idade e cluster predito) foram convertidos para Pandas (toPandas()).

Utilizou-se Matplotlib para gerar um histograma colorido, destacando a distribuição de idades e as fronteiras dos grupos identificados.

4. Pontos sobre o Modelo (K-Means no Spark)

Escalabilidade: O código foi desenhado para funcionar tanto em um dataset de 800 pilotos quanto em um de 800 milhões, graças à arquitetura distribuída do Spark.

Centróides: Os centros calculados pelo algoritmo representam a "idade média" de cada geração de pilotos.

Labels Automáticas: O modelo atribuiu automaticamente cada piloto a um grupo (0, 1 ou 2) sem necessidade de regras manuais ("se idade > 30...").

Insights Possíveis:

Identificar a renovação do grid (proporção de jovens vs. veteranos).

Analisar a longevidade dos pilotos em diferentes eras da F1.

5. Resultados

O modelo convergiu com sucesso e separou os pilotos em 3 faixas etárias distintas.

A análise visual confirmou que o agrupamento seguiu uma lógica cronológica (ex: pilotos atuais em um cluster, lendas históricas em outro).

O uso de VectorAssembler demonstrou a especificidade do pipeline de ML em Big Data.

6. Possíveis Melhorias

Método do Cotovelo (Elbow Method): Automatizar a escolha do k (número de clusters) calculando o erro quadrático (WSSSE) para vários valores.

Mais Features: Incluir "número de vitórias" ou "anos de atividade" para criar clusters baseados em performance e experiência, não apenas idade.

Persistência: Salvar o modelo treinado em disco para reutilização em pipelines de produção.

7. Conclusão

O projeto validou o uso do PySpark para tarefas de ponta a ponta, desde a ingestão de dados (ETL) até a aplicação de modelos de IA. A complexidade adicional de configuração foi compensada pela robustez e capacidade de processamento massivo da ferramenta.

--8<-- ".\scripts\ml_f1.py"
