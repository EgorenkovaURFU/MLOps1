from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import numpy as np

X_train = pd.read_csv('train/xtr_prep.csv', index_col=[0])
y_train = np.load('train/ytr.npy')


clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(X_train, y_train)

pickle.dump(clf, open('models/model.pkl', 'wb'))

