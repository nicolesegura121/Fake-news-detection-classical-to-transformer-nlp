# Fake-news-detection-classical-to-transformer-nlp

## Group Project

This project explores multiple Natural Language Processing (NLP) approaches for fake news classification, ranging from classical Bag of Words and TF-IDF methods to sentence embeddings and transformer-based models such as DistilBERT.

---

# Team Members

| Member | Model Focus | GitHub |
|---|---|---|
| Alan An Jung Wei | --- | [alananjungwei](https://github.com/alananjungwei) |
| Laxmi Gupte | --- | [laxmigs24](https://github.com/laxmigs24) |
| Nicole Segura | --- | [nicolesegura121](https://github.com/nicolesegura121) |

---

# Objectives

- Build a fake news classifier
- Compare classical NLP and transformer approahces
- Evaluate multiple machine learning models 
- Perform inference on unseen news data (the testing dataset)
- Analyze model performance and prediction behaviour

---

# Dataset

The dataset consists of news headlines/articles labeled as:
- 0 → fake news
- 1 → real news

The testing dataset contained placeholder labels (2), which were replaced using model predictions.

---

# Workflow Overview 
I improved the CNN architecture through multiple experiments, with step-by-step optimization.


# Data Cleaning Section 




# Classical NLP Section 

# Embedding Section 

# Transformer Section 


# Hyperparameter Tuning 



# Results Table 

| Model | Main Change | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| BOW + MultinomialNB | default| 0.93 | 0.93 | 0.94 | 0.93 | 
| BOW + MultinomialNB | n_gram (1, 2), max_df = 0.95, min_df = 2, max_features = 10000 | 0.93 | 0.93 | 0.93 | 0.93 | 
| BOW + MultinomialNB| n_gram (1, 2), max_df = 0.99, min_df = 1, max_features = 50000 | 0.94 | 0.94 | 0.94 | 0.94 | 
| BOW + MultinomialNB | n_gram (1, 2), max_df = 0.95, min_df = 2, max_features = 50000| 0.94 | 0.94 | 0.94 | 0.94 | 
| TF-IDF + MultinomialNB | n_gram (1, 2), max_df = 0.95, min_df = 2, max_features = 20000 | 0.93 | 0.93 | 0.94 | 0.93 | 
| TF-IDF + MultinomialNB | n_gram (1, 3), max_df = 0.95, min_df = 2, max_features = 50000 | 0.94 | 0.94 | 0.94 | 0.94 | 
| Sentence Transformer + Logistic Regression | default | 0.92 | 0.91 | 0.93 | 0.92 | 
| Sentence Transformer + Logistic Regression | C = 10 | 0.92 | 0.91 | 0.93 | 0.92 | 
| Sentence Transformer + LinearSVC| default | 0.93 | 0.91 | 0.94 | 0.93 | 
| Sentence Transformer + LinearSVC| C = 10 | 0.92 | 0.91 | 0.94 | 0.93 | 
| Sentence Transformer + LinearSVC| C = 100 | 0.92 | 0.91 | 0.94 | 0.93 | 
| DistilBERT | default | 0.9618 | 0.9474 | 0.9784 | 0.9626 | 

## Classical NLP
* Bag of Words (BOW) + MultinomialNB Classifier 
* TF-IDF + MultinomialNB Classifier 
## Embeddings
* SentenceTransformer (Model: all-MiniLM-L6-v2) + LogisticRegression Classifier
* SentenceTransformer (Model: all-MiniLM-L6-v2) + LinearSVC Classifier
## Transformer 
* DistilBERT (Distilbert-base-uncased)

# Visualization Section 

![Prediction Distribution](images/prediction_distribution.png)


# Conclusions 