from sklearn.preprocessing import StandardScaler
import pandas as pd


X_train = pd.read_csv('train/xtr.csv')
X_test = pd.read_csv('test/xtest.csv')

scaler = StandardScaler()
scaler.fit(X_train)

scaler.transform(X_train)
scaler.transform(X_test)

X_train.to_csv('train/xtr_prep.csv')
X_test.to_csv('test/xtest_prep.csv')

