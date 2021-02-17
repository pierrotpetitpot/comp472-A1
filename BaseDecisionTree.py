import re
from main import *


# outputs words that only contain letters
def getUniqueWords(textFile):
    # only match letters
    regex = re.compile("^[A-Za-z]+$")

    with open(textFile, encoding='utf-8') as infile:
        unique = sorted(set(infile.read().split()))

    # filtered only contains words with letters
    filtered = [i for i in unique if regex.match(i)]
    return filtered


# outputs a matrix where the columns are the attributes (unique words) and where the rows are the number of
# occurrence for each word in the review
def getOccurrences(uniqueWords, reviews, iteration):
    result = []
    if iteration is None:
        for row in reviews:
            occurrence = []
            for attribute in uniqueWords:
                count = row.count(attribute)
                occurrence.append(count)

            result.append(occurrence)
    else:
        reviews = reviews[:iteration]
        for row in reviews:
            occurrence = []
            for attribute in uniqueWords:
                count = row.count(attribute)
                occurrence.append(count)

            result.append(occurrence)

    return result


uniqueWords = []
occurrences = []

uniqueWords = getUniqueWords('all_sentiment_shuffled.txt')
occurrences = getOccurrences(uniqueWords, all_docs, 1)

print(occurrences)
