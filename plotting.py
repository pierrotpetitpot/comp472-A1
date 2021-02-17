import matplotlib.pyplot as plt 
import numpy as np 
def plot (all_labels):
    totalCount = len(all_labels)
    posCount=all_labels.count("pos")
    negCount = all_labels.count("neg")
    
    data = {"pos:{}".format(posCount):posCount,"neg:{}".format(negCount):negCount} 
    sentiments = list(data.keys()) 
    values = list(data.values()) 
    
    fig = plt.figure(figsize = (15, 10)) 
    
    plt.bar(sentiments, values, color ='blue',  
            width=[0.6,0.6], align='center') 
    
    plt.xlabel("Possible Sentiments\nTotal Count:{}".format(totalCount)) 
    plt.ylabel("Number of reviews for each Sentiment") 
    plt.title("Plot of Number of reviews for each type of sentiment") 
    plt.show() 