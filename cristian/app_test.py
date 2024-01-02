from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report



X, y  =  load_iris(return_X_y=True)
print(type(X))
print(type(y))
print(X.shape)
print(y.shape)

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

clf  =  RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Train score
print("Train score")
print(clf.score(X_train, y_train))

# Test score
print("Test score")
print(clf.score(X_test, y_test))

