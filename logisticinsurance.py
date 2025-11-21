import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\hhhhhh\python\datasets\insurance_data.csv")
print(df)
print(df.info())

inputs = df.drop("bought_insurance", axis=1)
outputs = df.drop("age", axis=1)
print(inputs)
print(outputs)

plt.scatter(inputs, outputs)
plt.xlabel("Age")
plt.ylabel("Bought Insurance")
plt.show()

x_train,x_test,y_train,y_test=train_test_split(inputs,outputs,test_size=0.8)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(x_test)

