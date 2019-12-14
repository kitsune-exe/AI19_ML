import pickle
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time

theDataset = pd.read_csv('TraData.csv', header=None, sep=',').values
theDataset = np.array(theDataset)

x = theDataset[:, 1:]
y = theDataset[:, 0]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=0.1)
print('*********************')
print('* By Neural Network *')
print('*********************')
starttime = time.time()
clf4 = MLPClassifier(activation='relu',
                     solver='adam', #adam, 39, 39, 74.4
                     max_iter=1000,
                     hidden_layer_sizes=(60, 60),#best: 50, 50?
                     random_state=None)
clf4 = clf4.fit(train_x, train_y)

print('Training Accuracy:\n', clf4.score(train_x, train_y))
print('Training data prediction:\n', clf4.predict(train_x))
print('Validation Accuracy:\n', clf4.score(valid_x, valid_y))
print('Validation data prediction:\n', clf4.predict(valid_x))
endtime = time.time()
print('Time Used:\n', endtime - starttime)

f = open('clf4_NNetwork.pickle', 'wb')
pickle.dump(clf4, f)
f.close()
