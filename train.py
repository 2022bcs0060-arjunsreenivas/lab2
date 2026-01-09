import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("dataset/winequality-white.csv",sep=";")
X = data.drop(['quality'],axis=1)
y = data['quality']
X_subset = X.drop(["alcohol", "sulphates"], axis=1)

X_train,X_test,y_train,y_test = train_test_split(X_subset,y,test_size=0.2, random_state=1234)

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

mse = mean_squared_error(pred,y_test)
r2 = r2_score(pred,y_test)

print("MSE: ",mse)
print("R2 score: ",r2)

joblib.dump(model,"output/winequality-linearmodel.pkl")
metrics = {
    "MSE" : mse,
    "R2_Score": r2,
}

with open("output/winequality-linearmodel_metrics.json","w") as file:
    json.dump(metrics,file)