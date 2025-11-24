import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load
df = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\hhhhhh\python\datasets\insurance_data.csv")

# Split inputs & outputs
inputs = df["age"]              # feature
outputs = df["bought_insurance"]  # target

# Plot
plt.scatter(inputs, outputs)
plt.xlabel("Age")
plt.ylabel("Bought Insurance")
plt.show()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2)

# Logistic Regression model
model = LogisticRegression()
model.fit(x_train.values.reshape(-1,1), y_train)

# Predict
y_pred = model.predict(x_test.values.reshape(-1,1))
print("Predicted:", y_pred)
print("Actual:", y_test.values)
