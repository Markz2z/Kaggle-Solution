import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

#titanic = pd.read_csv("train.csv")
titanic = pd.read_csv("train.csv", dtype={"Age": np.float64}, )

print(titanic.head(5))

print(titanic.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

#process sex
print(titanic["Sex"].unique())

titanic.loc[titanic["Sex"] == "male", "Sex"] = 1

titanic.loc[titanic["Sex"] == "female", "Sex"] = 0

#process Embarked
print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

#Linear Regression
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=4, random_state=1)

predictions = []
for train, test in kf:
	train_predictors = (titanic[predictors].iloc[train,:])
	train_target = titanic["Survived"].iloc[train]
	alg.fit(train_predictors, train_target)
	test_predictions = alg.predict(titanic[predictors].iloc[test, :])
	predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > 0.5] = 1
predictions[predictions < 0.5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print ("linear regression result")
print (accuracy)

#Logistic Regression

alg = LogisticRegression(random_state = 1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=4)
print("logistic regression result")
print(scores.mean())

#test
#feature engineering
titanic_test = pd.read_csv("test.csv", dtype={"Age": np.float64}, )
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 1
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 0
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

#submit
alg = LogisticRegression(random_state = 1)
alg.fit(titanic[predictors], titanic["Survived"])
predictions = alg.predict(titanic_test[predictors])

submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],
"Survived": predictions})

submission.to_csv("kaggle.csv", index=False)

