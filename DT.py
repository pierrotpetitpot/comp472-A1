from sklearn.tree import DecisionTreeClassifier
from results import write_stats

def dtRun (x_train,x_test,y_train,y_test,split):
    # define DT clf
    clf = DecisionTreeClassifier(criterion = 'entropy')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # write stats to file
    write_stats(y_pred, y_test, 'BaseDecisionTree-dataset.txt',split)