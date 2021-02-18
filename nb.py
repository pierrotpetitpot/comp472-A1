from sklearn.naive_bayes import MultinomialNB
from results import write_stats

#Defining the Naive Bayes Multinomial
def nbRun(x_train,x_test,y_train,y_test,split):
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    #Writing the stats 
    write_stats(y_pred, y_test, 'NaiveBayes-Dataset.txt',split)

