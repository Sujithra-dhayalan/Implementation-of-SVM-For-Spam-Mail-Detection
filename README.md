# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries
2. Read the dataset 
3. Split the dataset into training dataset and testing dataset
4. Train the model using SVM and print the output

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sujithra D
RegisterNumber:  212222220052
*/

import pandas as pd
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')
data.tail()
data.info()
data.isnull().sum()
x=data['v2'].values
y=data['v1'].values
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train.shape
x_test.shape
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/d7a2688f-c623-451e-a303-6f9b144fd369)
![image](https://github.com/user-attachments/assets/c9831dee-fc04-4642-b00d-703c666615d4)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
