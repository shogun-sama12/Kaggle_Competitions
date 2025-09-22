import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score,confusion_matrix,classification_report

df= pd.read_csv('./proceeded_titanic.csv')
y=df['Survived']
X=df.drop(['Survived','PassengerId'],axis=1)

lb=LabelEncoder()
categorial_cols=X.select_dtypes(include=['object']).columns
for col in categorial_cols:
    X[col]=lb.fit_transform(X[col])

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=.3,random_state=42)

model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred= model.predict(X_test)
print(y_pred[:30])

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("Classification Report:\n", classification_report(y_test, y_pred))

