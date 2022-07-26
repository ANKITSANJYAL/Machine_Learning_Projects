import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import datasets , linear_model
from sklearn.model_selection import train_test_split
data = datasets.load_diabetes()

X = data.data 
y = data.target 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .20)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
print('predictions:',prediction)
print('mean squared error:',mean_squared_error(y_test, prediction))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)