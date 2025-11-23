import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\hhhhhh\python\datasets\Social_Network_Ads.csv")

print(df["Gender"].unique())
print(df["Gender"].nunique())

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
print(df)
print(df.info())

inputs = df.drop(["User ID", "Purchased"], axis=1)
outputs = df.drop(["User ID", "Gender", "Age", "EstimatedSalary"], axis=1)
print(inputs)
print(outputs)

x_train,x_test,y_train,y_test = train_test_split(inputs, outputs, test_size=0.2)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_test)
