from.sklearn.naive_bayes import MultinomialNB
from data_helper import pre_process, plot, write_stats

#Pre-processing data
x_train, x_test, y_train, y_test = pre_process(open('all_sentiment_shuffled.txt', encoding="utf8"))

#Plotting the counts
plot(y_train + y_test)

#Defining the Naive Bayes Multinomial
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#Writing the stats 
write_stats(y_pred, y_test, 'NaiveBayes_Sentiment.txt')

