import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\hhhhhh\python\datasets\Housing.csv")
print(dataset.isnull().sum())

x = dataset.drop("price" , axis = 1)
y = dataset["price"]

label_encoder = LabelEncoder()
for column in x.select_dtypes(include=['object']).columns:
    x[column] = label_encoder.fit_transform(x[column])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

regressor = RandomForestRegressor(n_estimators= 50, random_state=0)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error", np.sqrt(metrics.root_mean_squared_error(y_test, y_pred)))