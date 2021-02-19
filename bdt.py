from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from results import write_stats


def bdtRun(x_train, x_test, y_train, y_test, split):
    # model = DecisionTreeClassifier()
    # param_dict = {
    #     "criterion" : ['gini', 'entropy'],
    #     'splitter' : ['gini', 'random'],
    #     'max_depth' : [5,10,100,1000],
    #     'min_samples_split' : [2,5,10],
    #     'min_samples_leaf' : [1,3,5],
    #     'max_features' : ['auto', 'sqrt', 'log2']
    # }

    # grid = RandomizedSearchCV(model, param_distributions=param_dict, verbose=1, n_jobs=1, n_iter=6, cv=2)
    # grid.fit(x_train, y_train)

    # print(grid.best_estimator_)
    # best_params = grid.best_params_

    # model = DecisionTreeClassifier(criterion=best_params['criterion'], \
    #                                 splitter=best_params['splitter'], \
    #                                 max_depth=best_params['max_depth'], \
    #                                 min_impurity_split=best_params['min_samples_split'], \
    #                                 min_samples_leaf=best_params['min_samples_leaf'],\
    #                                 max_features=best_params['max_features'])
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)

    # write_stats(y_pred, y_test, 'BestDecisionTree-dataset.txt',split)
    # define DT clf
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=15000,
            min_samples_leaf=2, min_samples_split=0.785,
            splitter='random')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # write stats to file
    write_stats(y_pred, y_test, 'BestDecisionTree-dataset.txt', split)
