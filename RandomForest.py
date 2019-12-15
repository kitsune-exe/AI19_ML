import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time

print('Read input files')
theDataset = pd.read_csv('TraData.csv', header=None, sep=',').values
theDataset = np.array(theDataset)

print('Cut x, y data into train/test')
x = theDataset[:, 1:]
y = theDataset[:, 0]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=0.9)

print('********************')
print('* By Random Forest *')
print('********************')
starttime = time.time()
clf2 = RandomForestClassifier(criterion='entropy',
                              n_estimators=390,
                              max_depth=7,
                              random_state=0,
                              max_features=None)
clf2 = clf2.fit(train_x, train_y)

print('***********')
print('* Results *')
print('***********')

print('Training Accuracy:\n', clf2.score(train_x, train_y))
print('Training data prediction:\n', clf2.predict(train_x))
print('Validation Accuracy:\n', clf2.score(valid_x, valid_y))
print('Validation data prediction:\n', clf2.predict(valid_x))
endtime = time.time()
print('Time used:\n', endtime - starttime)

f = open('clf2_RForest.pickle', 'wb')
pickle.dump(clf2, f)
f.close()
