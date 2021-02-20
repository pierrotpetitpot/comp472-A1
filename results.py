
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

def write_stats (predictedLabels, predictLabels, outFile, splitPoint):
    outFile = open (outFile,'w')

    #Displaying the confusion matrix with the appropriate text
    outFile.write ("Report from scikit learn:")
    outFile.write ("\n\nDisplaying the confusion matrix:\n\nTrue Negatives , False Positives\nFalse Negatives , True Positives ")
    outFile.write(f'\n\n{np.array2string(confusion_matrix(predictLabels,predictedLabels))}')

    #Displaying the analytics provided by scikit-learn methods
    outFile.write("\n\nOutputting all the metrics required (precision,recall, f1 score) and the count in each of the sentiments:")
    outFile.write(f'\n\n{classification_report(predictLabels,predictedLabels,target_names=["Neg","Pos"])}')

    #Displaying predictions with their line number
    outFile.write ("\nTable with prediction and the instance position:")
    outFile.write(''.join(f'\n{i}, {n}\n' for i, n in enumerate (predictedLabels,splitPoint)))
    outFile.close()
