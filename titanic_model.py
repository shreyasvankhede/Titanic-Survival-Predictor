import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def model(pclass,sex,age):
    df=pd.read_csv("Titanic-Dataset.csv")
    Y=df.iloc[:,1].values
    X=df.drop(["PassengerId","Name","SibSp",'Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
    le = LabelEncoder()
    X.iloc[:, 2] = le.fit_transform(X.iloc[:, 2])  # Assuming column 2 is 'Sex'
    X=X.drop(["Survived"],axis=1)
    X=X.values
    X_imp=SimpleImputer(missing_values=np.nan,strategy="mean")
    X=X_imp.fit_transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    regressor=LogisticRegression()
    regressor.fit(X_train,Y_train)
    np.set_printoptions(precision=2) #prints upto 2 decimals only
    return regressor.predict([[pclass,sex,age]])

def res(n):
  if n==0:
    return "You didnt survive the titanic"
  else:
    return "You survived the titanic"

def info():
  age=int(input("Enter the age: "))
  cl=int(input("Enter the passenger class no. (1-3): "))
  sex=int(input("Enter your gender 0.For Female 1.For male: "))
  while (age<0 or cl<1 or cl>3 or sex>1 or sex<0):
    if (sex>1 or sex<0):
      sex=int(input("Enter a valid number for gender \n1.Male \n0.Female: ") )
    elif(cl<1 or cl>3):
      cl=int(input("Enter a valid number for passenger class no. (1-3): "))
    elif (age<0):
      age=int(input("Enter a valid number for age (age cant be negative): "))
    else:
      break
  print(res(model(cl,sex,age)))

if __name__=="__main__":
    info()