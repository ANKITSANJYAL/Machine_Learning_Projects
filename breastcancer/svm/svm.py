import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data
y = iris.target

Classes = ['Iris Setosa', 'Iris Versicolour' ,'Iris Virginica']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2)

model = svm.SVC()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test , prediction)

print("prediction  :" , prediction)
print('actual value:' , y_test)
print("accuracy:" , accuracy)

for i in range(len(prediction)):
    print(Classes[prediction[i]])