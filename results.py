
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

def write_stats (y_pred, y_test, file, split):
    file = open (file,'w')

    file.write ("Report from scikit learn:")
    file.write(f'\n{classification_report(y_test,y_pred,target_names=["Neg","Pos"])}')


    file.write ("\n\nDisplaying the confusion matrix:")
    file.write(f'\n{np.array2string(confusion_matrix(y_test,y_pred))}')

    file.write ("\nTable with prediction and the instance position:")
    file.write(f'\n{i}, {n}\n' for i, n in enumerate (y_pred,split))
    file.close()