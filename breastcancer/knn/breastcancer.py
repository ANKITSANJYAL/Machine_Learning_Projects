import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn import neighbors , metrics
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

df = pd.DataFrame(np.c_[data.data , data.target] , columns = [list(data.feature_names)+['target']])

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .20)

knn = neighbors.KNeighborsClassifier(n_neighbors=25 , weights = 'uniform')
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test , prediction)

print("prediction:" , prediction)
print("accuracy:" , accuracy)