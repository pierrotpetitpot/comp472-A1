from __future__ import division
from codecs import open
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error
from nb import nb
from plotting import plot

docs = []
labels = []
with open("all_sentiment_shuffled.txt", encoding='utf-8') as f:
    for line in f:
        words = line.strip().split()
        docs.append(words[3:])
        labels.append(words[1])

all_docs = docs
all_labels = labels

#plot(all_labels)
nb(all_docs,all_labels)


