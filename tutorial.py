# Iris
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
train_x, valid_x, train_y, valid_y = train_test_split(
    iris.data, iris.target, test_size=0.1) #切割train跟test之占比
print(valid_x, valid_y)

#Decision Tree
print('\nDecision Tree\n')
clf1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)
clf1 = clf1.fit(train_x, train_y)
print('Training Accuracy:\n', clf1.score(train_x, train_y))
print('Training data prediction:\n', clf1.predict(train_x))
print('Validation Accuracy:\n', clf1.score(valid_x, valid_y))
print('Validation data prediction:\n', clf1.predict(valid_x))

#Random Forest
print('\nRandom Forest\n')
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(criterion='entropy',
                            n_estimators=500,
                            max_depth=7,
                            random_state=0,
                            max_features='log2')
clf2 = clf2.fit(train_x, train_y)
print('Training Accuracy:\n', clf2.score(train_x, train_y))
print('Training data prediction:\n', clf2.predict(train_x))
print('Validation Accuracy:\n', clf2.score(valid_x, valid_y))
print('Validation data prediction:\n', clf2.predict(valid_x))

# Nearest Neighbor
print('\nNearest Neighbors\n')
from sklearn.neighbors import KNeighborsClassifier
clf3 = KNeighborsClassifier(n_neighbors=5)
clf3 = clf3.fit(train_x, train_y)
print('Training Accuracy:\n', clf3.score(train_x, train_y))
print('Training data prediction:\n', clf3.predict(train_x))
print('Validation Accuracy:\n', clf3.score(valid_x, valid_y))
print('Validation data prediction:\n', clf3.predict(valid_x))

# Neural Network
print('\nNeural Network\n')
from sklearn.neural_network import MLPClassifier
clf4 = MLPClassifier(activation='relu',
                     solver='sgd',
                     max_iter=2000,
                     hidden_layer_sizes=(5, 5),
                     random_state=None)
clf4 = clf4.fit(train_x, train_y)
print('Training Accuracy:\n', clf4.score(train_x, train_y))
print('Training data prediction:\n', clf4.predict(train_x))
print('Validation Accuracy:\n', clf4.score(valid_x, valid_y))
print('Validation data prediction:\n', clf4.predict(valid_x))

# Save Model:
import pickle

f = open('clf1.pickle', 'wb')
pickle.dump(clf1, f)
f.close()

# Load Model:
f = open('clf1.pickle', 'rb')
trained_clf1 = pickle.load(f)
f.close()
print('\nTrained model\'s prediction:\n', trained_clf1.predict(valid_x))

# Load CSV
"""
import pandas as pd
df = pd.read_csv('TraData.csv', header=None)
print('\n', df)
"""