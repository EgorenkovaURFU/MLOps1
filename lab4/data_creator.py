from catboost.datasets import titanic

train, test = titanic()

train.to_csv('lab4/datasets/data.csv')

