import sklearn.metrics as sm
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups as fz
from sklearn.feature_extraction.text import CountVectorizer

train_data=fz(subset="train",shuffle=True)
categories=train_data.target_names[:4]
print(categories)
train_data=fz(subset="train",categories=categories,shuffle=True)
test_data=fz(subset="test",categories=categories,shuffle=True)

cv=CountVectorizer()
X_train=cv.fit_transform(train_data.data)

model=MultinomialNB()
model.fit(X_train,train_data.target)

X_test=cv.transform(test_data.data)
y_pred=model.predict(X_test)

print("\nAccuracy",sm.accuracy_score(test_data.target,y_pred)*100)
print("\nConfusion Matrix:\n",sm.confusion_matrix(test_data.target,y_pred))
print("\nClassification Report: \n",sm.classification_report(test_data.target,y_pred,target_names=categories))
