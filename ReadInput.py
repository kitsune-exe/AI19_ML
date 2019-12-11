import pandas as pd
import pickle
import numpy as np
#import sklearn
infile = pd.read_csv('input.csv', header=None).values
infile = np.array(infile)

f = open('clf4_NNetwork.pickle', 'rb')
trained_cf = pickle.load(f)
f.close()
np.savetxt('output.csv', trained_cf.predict(infile), fmt='%d')
