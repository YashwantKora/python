# 1️⃣ Import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 2️⃣ Load
df = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\hhhhhh\python\datasets\homeprices.csv")

# 3️⃣ Prepare
X = df[['area']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 4️⃣ Model
model = LinearRegression()

# 5️⃣ Train/Test
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Actual:", y_test.values)
print("MSE:", mean_squared_error(y_test, y_pred))
