import matplotlib.pyplot as plt 
import numpy as np 
def plot (trainingLabels):

    #Assembling the data for the bar graph   
    totalCount = len(trainingLabels)
    posCount=trainingLabels.count(1)
    negCount = trainingLabels.count(0)

    #Formatting the data for the bar graph
    data = {"pos:{}".format(posCount):posCount,"neg:{}".format(negCount):negCount} 
    sentiments = list(data.keys()) 
    values = list(data.values()) 
    
    #fitting the window size
    fig = plt.figure(figsize = (15, 10)) 
    
    #building the plot 
    plt.bar(sentiments, values, color ='blue',  
            width=[0.6,0.6], align='center') 
    
    #labeling and printing the plots
    plt.xlabel("Possible Sentiments\nTotal Count:{}".format(totalCount)) 
    plt.ylabel("Number of reviews for each Sentiment") 
    plt.title("Plot of Number of reviews for each type of sentiment") 
    plt.show() 