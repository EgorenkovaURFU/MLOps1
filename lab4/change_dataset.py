import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r'lab4/datasets/data.csv', index_col=[0])
# df = df[['Pclass', 'Sex', 'Age']]
# df.to_csv(r'lab4/datasets/data.csv')

# df['Age'] = df['Age'].fillna(df['Age'].mean())
# df.to_csv(r'lab4/datasets/data.csv')

cat_cal = ['Sex']
oh = OneHotEncoder(sparse_output=False)
oh_sex = oh.fit_transform(df[cat_cal])

df_sex = pd.DataFrame(oh_sex, columns=oh.get_feature_names_out(cat_cal))

df = df.drop(columns=['Sex'])
df = pd.concat([df, df_sex], axis=1)
df.to_csv(r'lab4/datasets/data.csv')

