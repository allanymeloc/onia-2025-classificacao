# ONIA 2025 — Classificação Multiclasse com Dados Tabulares

Projeto desenvolvido para a Prova de Análise de Dados da Olimpíada Nacional de Inteligência Artificial (ONIA 2025), na qual obtive **medalha de bronze** e classificação até a **4ª fase**.

O desafio consistia em resolver um problema de **classificação multiclasse** (5 classes) a partir de um conjunto de dados tabulares, comparando diferentes algoritmos de aprendizado supervisionado e ajustando hiperparâmetros para maximizar o **F1-score**.

## 🎯 Objetivo

- Comparar modelos de classificação (Decision Tree, Random Forest, Gradient Boosting, SVM, KNN e MLP)
- Realizar tuning de hiperparâmetros com GridSearchCV
- Selecionar o modelo com melhor F1-score em validação
- Gerar as predições finais para o conjunto de teste

## 🧠 Metodologia

1. Carregamento e inspeção dos dados (treino.csv e teste.csv)
2. Split treino/validação (90/10, random_state=42)
3. Padronização com StandardScaler
4. Comparação de modelos via validação cruzada (10-fold)
5. Tuning com GridSearchCV nos melhores modelos (KNN, Random Forest e MLP)
6. Modelo final: MLPClassifier (rede neural) — melhor F1-score ponderado
7. Predição no conjunto de teste e geração do arquivo de submissão

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

## 🤖 O Modelo Escolhido

**Multi-Layer Perceptron (MLPClassifier)**

A escolha se deu pelo melhor desempenho em F1-score ponderado no conjunto de validação (0.84), superando os modelos baseados em árvores e o KNN. O MLP é capaz de modelar relações não lineares complexas, o que se mostrou vantajoso para este conjunto de dados.

## 🛠️ Tecnologias

- Python 3.10+
- scikit-learn
- pandas / NumPy
- Matplotlib
- Google Colab

## 👩‍💻 Autora

**Allany de Melo Camacho**  
Estudante de Estatística — UFF  

## 📌 Contexto

- 🥉 Medalha de Bronze — ONIA 2025 (classificada até a 4ª fase)
- 🌍 Participante da IOAI 2024 (Burgas, Bulgária)
