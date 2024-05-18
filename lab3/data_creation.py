from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


full_data = load_iris()

data = full_data.data
feature = full_data.feature_names

print(full_data.target_names)

X = pd.DataFrame(data=data, columns=feature)
y = full_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.to_csv('train/xtr.csv')
np.save('train/ytr.npy', y_train)
X_test.to_csv('test/xtest.csv')
np.save('test/ytest.npy', y_test)
