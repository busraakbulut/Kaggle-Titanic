from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ids = test["PassengerId"]


def null_values(train):
    train = train.drop(["Ticket", "PassengerId", "Name", "Cabin"], axis=1)

    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        train[col].fillna(train[col].median(), inplace=True)

    train.Embarked.fillna("U", inplace=True)
    return train


train = null_values(train)
test = null_values(test)
le = preprocessing.LabelEncoder()
columns = ["Sex", "Embarked"]

for col in columns:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    print(le.classes_)

y = train["Survived"]
X = train.drop("Survived", axis=1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression(
    random_state=0, max_iter=1000).fit(X_train, y_train)
log_reg_predict = log_reg.predict(X_val)
print(accuracy_score(y_val, log_reg_predict))

submission_preds = log_reg.predict(test)
df = pd.DataFrame({"PassengerId": test_ids.values,
                   "Survived": submission_preds,
                   })
df.to_csv("submission1.csv", index=False)
