import pickle
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import time
theDataset = pd.read_csv('TraData.csv', header=None, sep=',').values
theDataset = np.array(theDataset)

x = theDataset[:, 1:]
y = theDataset[:, 0]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=0.75)
#print(x, train_x, valid_x)

print('********************')
print('* By Decision Tree *')
print('********************')
starttime = time.time()
clf1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
clf1 = clf1.fit(train_x, train_y)
print('Training Accuracy:\n', clf1.score(train_x, train_y))
print('Training data prediction:\n', clf1.predict(train_x))
print('Validation Accuracy:\n', clf1.score(valid_x, valid_y))
print('Validation data prediction:\n', clf1.predict(valid_x))
endtime = time.time()
print('Time used:\n', endtime - starttime)
# Save Model:
f = open('clf1_DTree.pickle', 'wb')
pickle.dump(clf1, f)
f.close()
np.savetxt('output_DT.csv', clf1.predict(valid_x), fmt='%d')
