import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import neighbors , metrics

data= load_wine()

df = pd.DataFrame(np.c_[data.data, data.target], columns=[list(data.feature_names)+['target']])


X = df.iloc[:,0:-1]
y = df.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2)

knn = neighbors.KNeighborsClassifier(n_neighbors=37)
knn.fit(X_train, y_train)

predictor = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test,predictor)

print("prediction:" , predictor)
print("accuracy:" , accuracy)