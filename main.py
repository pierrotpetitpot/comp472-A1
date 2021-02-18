from __future__ import division
from codecs import open
import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np
from plotting import plot
import string
import collections
from bdt import bdtRun
from nb import nbRun

split = 0
# we instantiate the matrix with these numbers for performance
wordcount = 54090
instancescount = 11914

labels = []
# features is an empty matrix where each row is the number of occurrence of an unique word
# in the review
features = [[0]*wordcount for _ in range(instancescount)]
word_hash = collections.defaultdict(int)

# i is the count of unique words
i=0

with open("all_sentiment_shuffled.txt", encoding='utf-8') as f:
# we go line by line
    for j, line in enumerate(f):
        # we split the line by words
        line = line.split()
        if line[1]=='pos':
            labels.append(1)
        else:
            labels.append(0)

        #line is the actual review
        line = line [3:]
        for w in line:
            w= w.translate (str.maketrans('', '', string.punctuation))
        if w:
            #if the word is unique, we add it to the hash
            if w not in word_hash:
                features[j][i] +=1
                word_hash[w] = i
                i += 1
            else:
                # else we just add a count
                features[j][word_hash[w]] +=1

f.close()
split = int(0.8 * len(labels))

x_train = features[:split]
x_test = features[split:]
y_train = labels[:split]
y_test = labels[split:]

print (len(x_train))
print (len(x_test))
print (len(y_train))
print (len(y_test))

#plot(y_train)
nbRun(x_train,x_test,y_train,y_test,split)
#dtRun(x_train,x_test,y_train,y_test,split)
#bdtRun(x_train,x_test,y_train,y_test,split)