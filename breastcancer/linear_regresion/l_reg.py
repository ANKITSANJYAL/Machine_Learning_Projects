from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model

import matplotlib.pyplot as plt

data = datasets.load_boston()
X = data.data
y = data.target

plt.scatter(X.T[0],y)


X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2)

l_reg = linear_model.LinearRegression()

model = l_reg.fit(X_train,y_train)

prediction = model.predict(X_test)


print('prediction   :',prediction)
print("actual_value :",y_test)

print("R^2" , l_reg.score(X,y))
print("coeff:" , l_reg.coef_)
print("intercept:",l_reg.intercept_)