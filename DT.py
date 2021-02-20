from sklearn.tree import DecisionTreeClassifier
from results import write_stats

def dtRun (trainingDocs,predictDocs,trainingLabels,predictLabels,splitPoint):
    
    #buidling the model with the appropriate criterion
    model = DecisionTreeClassifier(criterion = 'entropy')
    #fitting the model with the training data set    
    model.fit(trainingDocs, trainingLabels)
    #getting the predicted labels from the model
    predictedLabels = model.predict(predictDocs)

    #calling the output function to display the analytics    
    write_stats(predictedLabels, predictLabels, 'BaseDecisionTree-dataset.txt',splitPoint)