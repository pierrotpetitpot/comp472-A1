from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from results import write_stats
#Defining the Naive Bayes Multinomial
def nbRun(trainingDocs,predictDocs,trainingLabels,predictLabels,split):

    #building the model with the best found algorigthm(Multinomial)
    model = MultinomialNB()
    
    #fitting the model with the training data set    
    model.fit(trainingDocs, trainingLabels)
    
    #getting the predicted labels from the model
    predictedLabels = model.predict(predictDocs)
    
    #calling the output function to display the analytics    
    write_stats(predictedLabels, predictLabels, 'NaiveBayes-Dataset.txt',split)

