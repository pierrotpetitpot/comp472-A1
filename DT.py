from sklearn.tree import DecisionTreeClassifier
from main import *

# Pre process/Extract data
x_train, x_test, y_train, y_test = pre_process(open('all_sentiment_shuffled.txt', encoding='utf8'))

# plot the class counts
plot(y_train + y_test)

# define DT clf
clf = DecisionTreeClassiFier(criterion = 'entropy')
clf.fit(x_train, y_train)
y_pref = cld.predict(x_test)

# write stats to file
write_stats(y_prod, y_test, 'DT.txt')