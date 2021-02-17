from __future__ import division
from codecs import open
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


def nb(all_docs,all_labels):
    split_point = int(0.80*len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]
    
    flatten_train_docs =[' '.join(ele) for ele in train_docs]
    flatten_eval_docs =[' '.join(elem) for elem in eval_docs]

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(flatten_train_docs)
    X_eval_counts = count_vect.fit_transform(flatten_eval_docs)
    print (X_train_counts.shape)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_eval_tfidf = tfidf_transformer.fit_transform(X_eval_counts)
    print(X_train_tfidf.shape)

    model = GaussianNB().fit(X_train_tfidf.todense(), train_labels)

    predicted = model.predict(X_eval_tfidf.todense())
    print (predicted)

