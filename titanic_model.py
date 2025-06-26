import pandas as pd
import numpy as np
import gradio as gr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class TitanicPred:
  def __init__(self):
      self.data="Data/Titanic-Dataset.csv"
      self.reg=self.train_model()
      self.gif={
         0:"Data/Images/0.gif",
         1:"Data/Images/1.gif"
      }

  def train_model(self):
    df=pd.read_csv(self.data)
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
    return regressor
  
  def predict(self,pclass,sex,age):
     return self.reg.predict([[pclass,sex,age]])

  def info(self,cl,sex,age):
   res=self.predict(cl,sex,age)
   img=self.gif[res[0]]
   return ("Hooraay !!! You would've survived the titanic" if res else "You would've not survived in the titanic"),img

def display(obj:TitanicPred):
    interface=gr.Interface(fn=obj.info,
                           inputs=[
                             gr.Slider(1,3,value=2,label="Class Number"),
                             gr.Radio([("Male",1),("Female",0)],label="Gender"),
                             gr.Slider(0,100,value=28,label="Enter your age")
                                   ],
                           outputs=[gr.Textbox(lines=2,type="text",placeholder="Output"),
                                    gr.Image(type='filepath',label="Survival status")
                           ],
                           title="Titanic Survival Predictor",
                           description="Enter your prefered class number,age and gender to see if you woudlve survived the titanic"
                           )
    interface.launch()


if __name__=="__main__":
    obj=TitanicPred()
    display(obj)