import numpy as np
import pandas as pd
from ML.model_selection import train_test_split
from ML.LinearRegression import LinearRegression
from ML.metrics import *


data = pd.read_csv("housing.txt", sep="\s+", header=None) # 使用正则表达式进行空格符分割
X = np.array(data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]) 

y = np.array(data.iloc[:,[13]])
y = y.reshape([506,])

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1250)

reg = LinearRegression()
reg.fit_normal(X_train, y_train)

y_predict = reg.predict(X_test)

print("均方误差为:", mean_squared_error(y_test, y_predict))
print("均方根误差为:", root_mean_squared_error(y_test, y_predict))
print("平均绝对误差", mean_absolute_error(y_test, y_predict))
print("R Squared为:", r2_score(y_test, y_predict))