import pandas as pd
from sklearn.tree import DecisionTreeClassifier

titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')

titanic_train = titanic_train.dropna()
X_train = titanic_train[['Age']]
y_train = titanic_train[['Survived']]

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


titanic_test = titanic_test.dropna()

X_test = titanic_test[['Age']]

y_predic = dtc.predict(X_test)

print(y_predic)

output = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': y_predic})
output.to_csv('my_submission.csv', index=False)