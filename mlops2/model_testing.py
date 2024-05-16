import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


X_test = pd.read_csv('test/xtest_prep.csv')
y_test = np.load('test/ytest.npy')


model = pickle.load(open('models/model.pkl', 'rb'))

pred = model.predict(X_test)

print(accuracy_score(y_test, pred))
