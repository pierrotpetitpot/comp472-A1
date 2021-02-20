from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from results import write_stats


def bdtRun(trainingDocs, predictDocs, trainingLabels, predictLabels, splitPoint):

    #building the model with the appropriate best attributes    
    model = DecisionTreeClassifier(criterion='entropy', max_depth=15000,
            min_samples_leaf=2, min_samples_split=0.785,
            splitter='random')
    #fitting the model with the training data set    
    model.fit(trainingDocs, trainingLabels)

    #getting the predicted labels from the model
    predictedLabels = model.predict(predictDocs)

    #calling the output function to display the analytics    
    write_stats(predictedLabels, predictLabels, 'BestDecisionTree-dataset.txt', splitPoint)
