# ONIA 2025 — Classificação Multiclasse com Dados Tabulares

Projeto desenvolvido para a **Prova de Análise de Dados da Olimpíada Nacional de Inteligência Artificial (ONIA 2025)**, na qual obtive **medalha de bronze** e classificação até a **4ª fase**.

O desafio consistia em resolver um problema de **classificação multiclasse** (5 classes) a partir de um conjunto de dados tabulares, comparando diferentes algoritmos de aprendizado supervisionado e ajustando hiperparâmetros para maximizar o **F1-score**.

## 🎯 Objetivo

- Comparar modelos de classificação (Decision Tree, Random Forest, Gradient Boosting, SVM, KNN e MLP)
- Realizar tuning de hiperparâmetros com `GridSearchCV`
- Selecionar o modelo com melhor F1-score em validação
- Gerar as predições finais para o conjunto de teste

## 🧠 Metodologia

1. **Carregamento e inspeção** dos dados (`treino.csv` e `teste.csv`)
2. **Split treino/validação** (90/10, `random_state=42`)
3. **Padronização** com `StandardScaler`
4. **Comparação de modelos** via validação cruzada (10-fold)
5. **Tuning** com `GridSearchCV` nos melhores modelos (KNN, Random Forest e MLP)
6. **Modelo final:** `MLPClassifier` (rede neural) — melhor F1-score ponderado
7. **Predição** no conjunto de teste e geração do arquivo de submissão

## 📊 Resultados

Abaixo estão os resultados obtidos no conjunto de validação (10% dos dados de treino) após o ajuste de hiperparâmetros:

| Modelo | F1-score (weighted) |
|---|---|
| Decision Tree (Gini) | 0.50 |
| Gradient Boosting | 0.55 |
| SVM (RBF) | 0.50 |
| Random Forest | 0.74 |
| KNN (k=7, euclidean) | 0.78 |
| **MLP (Modelo Final)** | **0.84** |

### Matriz de Confusão (MLP)
*A matriz de confusão do modelo final pode ser visualizada ao final do notebook, demonstrando o bom desempenho na maioria das classes.*

## 🤖 O Modelo Escolhido

**Multi-Layer Perceptron (MLPClassifier)** com a seguinte configuração:

```python
MLPClassifier(
    hidden_layer_sizes=(200,),
    max_iter=500,
    solver='adam',
    activation='relu',
    alpha=0.0001,
    learning_rate='constant',
    batch_size='auto',
    random_state=42
)
