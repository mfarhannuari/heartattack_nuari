import pickle

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# import dataset
df = pd.read_csv('heart.csv')
print(df)

y = df["output"]
x = df.drop(['output'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#Make pickle
pickle.dump(model, open("model1.pkl", "wb"))

# print(x)