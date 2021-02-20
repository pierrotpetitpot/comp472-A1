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
from dt import dtRun
from bdt import bdtRun

splitPoint = 0

with open("all_sentiment_shuffled.txt", encoding='utf-8') as inFile:

    #Limiting the array size for performance and building the array/word hash
    wordCount = 55000
    instancesCount = 11914 
    allDocs = [[0]*wordCount for _ in range(instancesCount)]
    wordHash = collections.defaultdict(int) 
    allLabels = []
    i=0

    #outer for loop iterating over each line in the file
    for j, line in enumerate(inFile):
        # we split the line by words
        line = line.split()
        #getting the sentiment and appending it to its array
        if line[1]=='neg':
            allLabels.append(0)
        else:
            allLabels.append(1)
        #discarding the unwanted data and keeping the correct words
        line = line [3:]
        #inner for loop iterating over each word
        for currentWord in line:
            #removing non english words, numerical data and punctuation
            currentWord= currentWord.translate (str.maketrans('', '', string.punctuation))
            currentWord= ''.join([i for i in currentWord if not i.isdigit()])
        if currentWord:
            #Evaluating if current word was already present in another sentence    
            if currentWord not in wordHash:
                #if it was not we append to the word hash and increment its count within the sentence
                allDocs[j][i] +=1
                wordHash[currentWord] = i
                i += 1
            else:
                # else we just increment its count within the sentence
                allDocs[j][wordHash[currentWord]] +=1

inFile.close()
#splitting the data between the training and evaluating data
split = int(0.8 * len(allLabels))

#assigning each data set to its desired variables
trainingDocs = allDocs[:split]
predictDocs = allDocs[split:]
trainingLabels = allLabels[:split]
predictLabels = allLabels[split:]

#plotting the training data
plot(trainingLabels)

#running the three classifiers
nbRun(trainingDocs,predictDocs,trainingLabels,predictLabels,split)
dtRun(trainingDocs,predictDocs,trainingLabels,predictLabels,split)
bdtRun(trainingDocs,predictDocs,trainingLabels,predictLabels,split)