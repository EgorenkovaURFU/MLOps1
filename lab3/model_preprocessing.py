from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle


X_train = pd.read_csv('train/xtr.csv', index_col=[0])
X_test = pd.read_csv('test/xtest.csv', index_col=[0])

scaler = StandardScaler()
scaler.fit(X_train)

scaler.transform(X_train)
scaler.transform(X_test)

X_train.to_csv('train/xtr_prep.csv')
X_test.to_csv('test/xtest_prep.csv')
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))


