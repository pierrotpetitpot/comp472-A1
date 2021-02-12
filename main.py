from __future__ import division
from codecs import open
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error

docs = []
labels = []
with open("all_sentiment_shuffled.txt", encoding='utf-8') as f:
    for line in f:
        words = line.strip().split()
        docs.append(words[3:])
        labels.append(words[1])

all_docs = docs
all_labels = labels

split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]



print (len(train_labels))
print (all_docs[0])
print(all_labels[0])



