# SVD Classification (lecture2/session01)

Pipeline: clean → tokenize → stopwords → synonyms → lemmatize → TF-IDF → TruncatedSVD (LSA) → normalization → LogisticRegression.

Dataset: scikit-learn 20 Newsgroups (4 classes; same subset as bowClassification).
Cache: C:\Users\murth\Desktop\NLP\cache\sk_cache

Outputs: model (joblib), confusion matrix PNG, predict script, README, minimal notebook.
