#step 1: import all necessary packages
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from pathlib import Path



#step 2: read csv file
data=Path("dataset/diabetes_risk_prediction_dataset.csv")
df=pd.read_csv(data)

#step 3: Performing EDA

print(df.shape) #(rows,cols)

print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.duplicated().sum())

#step 4: split and train, test
#split (features and target)
x=df.drop("class",axis=1) #axis 1-->representing column
y=df["class"] #target


#train and test
x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    stratify = y
)
x=pd.get_dummies(x,dtype=int) 

#training
model = LogisticRegression()
x_train = pd.get_dummies(x_train, drop_first=True)
x_test = pd.get_dummies(x_test, drop_first=True)

x_train, x_test = x_train.align(
x_test, join="left", axis=1, fill_value=0
)
model.fit(x_train,y_train)
#testing
x_test_pred=model.predict(x_train)
training_acc=accuracy_score(x_test_pred,y_train)

print(training_acc)
joblib.dump(model,"model.joblib")
joblib.dump(list(x.columns),"feature_names.joblib")
print("Model and feature names saved successfully!")

