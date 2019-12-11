from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time

theDataset = pd.read_csv('TraData.csv', header=None, sep=',').values
theDataset = np.array(theDataset)

x = theDataset[:, 1:]
y = theDataset[:, 0]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=0.9)

print('************************')
print('* By Nearest Neighbors *')
print('************************')
starttime = time.time()
clf3 = KNeighborsClassifier(n_neighbors=5)
clf3 = clf3.fit(train_x, train_y)
print('Training Accuracy:\n', clf3.score(train_x, train_y))
print('Training data prediction:\n', clf3.predict(train_x))
print('Validation Accuracy:\n', clf3.score(valid_x, valid_y))
print('Validation data prediction:\n', clf3.predict(valid_x))
endtime = time.time()
print('Time Used:\n', endtime - starttime)

f = open('clf3_NNeighbors.pickle', 'wb')
pickle.dump(clf3, f)
f.close()
