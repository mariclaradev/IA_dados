# Importar bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Carregar o dataset
data = pd.read_csv("Dataset.csv")

# Análise Exploratória dos Dados (EDA)
print("Resumo inicial dos dados:")
print(data.info())
print(data.describe())

# Verificar valores ausentes
data.isnull().sum()

# Substituir valores ausentes
print("Tratando valores ausentes...")
data['Idade'].fillna(data['Idade'].mean(), inplace=True)
data['Renda Anual (em $)'].fillna(data['Renda Anual (em $)'].median(), inplace=True)
data['Gênero'].fillna(data['Gênero'].mode()[0], inplace=True)
data['Anúncio Clicado'].fillna(data['Anúncio Clicado'].mode()[0], inplace=True)

# Corrigir valores inválidos no Tempo no Site
data['Tempo no Site (min)'] = data['Tempo no Site (min)'].apply(lambda x: max(x, 0))

# Codificação de variáveis categóricas
label_encoder = LabelEncoder()
data['Gênero'] = label_encoder.fit_transform(data['Gênero'])
data['Anúncio Clicado'] = label_encoder.fit_transform(data['Anúncio Clicado'])

# Visualizações para entender o dataset
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlações entre variáveis")
plt.show()

# Divisão em variáveis independentes e dependentes
X = data.drop(columns=['Compra (0 ou 1)'])
y = data['Compra (0 ou 1)']

# Normalização de variáveis numéricas
scaler = StandardScaler()
X[['Idade', 'Renda Anual (em $)', 'Tempo no Site (min)']] = scaler.fit_transform(
    X[['Idade', 'Renda Anual (em $)', 'Tempo no Site (min)']]
)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balanceamento de classes com SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Construção de Modelos
# Modelo 1: Regressão Logística
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_resampled, y_resampled)
logistic_preds = logistic_model.predict(X_test)

# Modelo 2: Árvore de Decisão
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_resampled, y_resampled)
tree_preds = tree_model.predict(X_test)

# Modelo 3: Random Forest
forest_model = RandomForestClassifier(random_state=42, n_estimators=100)
forest_model.fit(X_resampled, y_resampled)
forest_preds = forest_model.predict(X_test)

# Avaliação dos Modelos
def evaluate_model(name, y_test, y_pred):
    print(f"\nRelatório de Classificação - {name}")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}\n")

# Avaliar todos os modelos
evaluate_model("Regressão Logística", y_test, logistic_preds)
evaluate_model("\u00c1rvore de Decisão", y_test, tree_preds)
evaluate_model("Random Forest", y_test, forest_preds)

# Extra: Validação cruzada
logistic_cv = cross_val_score(logistic_model, X_resampled, y_resampled, cv=5).mean()
tree_cv = cross_val_score(tree_model, X_resampled, y_resampled, cv=5).mean()
forest_cv = cross_val_score(forest_model, X_resampled, y_resampled, cv=5).mean()

print("\nValidação Cruzada:")
print(f"Regressão Logística: {logistic_cv:.2f}")
print(f"\u00c1rvore de Decisão: {tree_cv:.2f}")
print(f"Random Forest: {forest_cv:.2f}")

# Visualizações de Importância de Features no Random Forest
importances = forest_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances, color='skyblue')
plt.title("Importância das Features (Random Forest)")
plt.xlabel("Importância")
plt.ylabel("Features")
plt.show()

# Carregando o dataset (substitua 'dataset.csv' pelo nome do arquivo correto)
data = pd.read_csv('dataset.csv')

# -----------------------------------
# Etapa 2: Pré-processamento dos Dados
# -----------------------------------

# 1. Verificando dados ausentes
print("Valores ausentes por coluna:")
print(data.isnull().sum())

# 2. Codificação de variáveis categóricas (Gênero e Anúncio Clicado)
data['Gênero'] = LabelEncoder().fit_transform(data['Gênero'])
data['Anúncio Clicado'] = LabelEncoder().fit_transform(data['Anúncio Clicado'])

# 3. Normalização das variáveis numéricas
scaler = StandardScaler()
data[['Idade', 'Renda Anual (em $)', 'Tempo no Site (min)']] = scaler.fit_transform(data[['Idade', 'Renda Anual (em $)', 'Tempo no Site (min)']])

# 4. Separação entre variáveis independentes (X) e dependente (y)
X = data.drop(['Compra (0 ou 1)'], axis=1)
y = data['Compra (0 ou 1)']

# 5. Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------
# Etapa 3: Construção do Modelo de Classificação
# -----------------------------------

# 1. Treinando o modelo de Árvore de Decisão
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 2. Avaliando o modelo
# Predições
y_pred = model.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Acurácia do modelo:", accuracy)
print("\nMatriz de Confusão:\n", conf_matrix)
print("\nRelatório de Classificação:\n", class_report)

# -----------------------------------
# Etapa 4: Interpretação dos Resultados
# -----------------------------------

# Identificando as variáveis mais importantes
feature_importances = model.feature_importances_
features = X.columns

# Visualização das importâncias
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features, palette="viridis")
plt.title("Importância das Variáveis no Modelo de Árvore de Decisão")
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.show()

# Sugestão de melhorias
print("\nPossíveis Melhorias:")
print("1. Testar outros modelos, como Random Forest ou Regressão Logística, para comparar desempenho.")
print("2. Realizar validação cruzada para uma avaliação mais robusta.")
print("3. Ajustar hiperparâmetros da Árvore de Decisão para melhorar a performance.")
