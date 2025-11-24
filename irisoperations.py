# 1️⃣ Import
import pandas as pd
import matplotlib.pyplot as plt

# 2️⃣ Load dataset
df = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\hhhhhh\python\datasets\Irisds.csv")
df.columns = ['Id','sepal_length','sepal_width','petal_length','petal_width','Species']

# 3️⃣ First 5 rows
print(df.head())

# 4️⃣ Mean & Standard Deviation
print("Column Means:")
print(df.mean(numeric_only=True))

print("Column Standard Deviations:")
print(df.std(numeric_only=True))

# 5️⃣ Display all functions/attributes
print("Functions in dataframe:")
print(dir(df))

# 6️⃣ Scatter plot (simple, clean)
plt.scatter(df['sepal_length'], df['sepal_width'], color='red')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width")
plt.show()
