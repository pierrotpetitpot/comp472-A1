from __future__ import division
from codecs import open
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error
from nb import nb
from plotting import plot

split = 0
wordcount = 54090
instancescount = 11914

# docs = []
# labels = []
# with open("all_sentiment_shuffled.txt", encoding='utf-8') as f:
#     for line in f:
#         words = line.strip().split()
#         docs.append(words[3:])
#         labels.append(words[1])

# all_docs = docs
# all_labels = labels


#plot(all_labels)
#nb(all_docs,all_labels)

labels = []
features [[0]*wordcount for _ in range(instancescount)]

i=0
for j, line in enumerate(open("all_sentiment_shuffled.txt", encoding='utf-8')):
    line = line.split()
    if line[1]=='pos':
        labels.append(1)
    else:
        labels.append(0)
    
    line = line [3:]
    for w in line:
        w= w.translate (str.maketrans('', '', string.punctuation))

