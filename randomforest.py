# 1️⃣ Import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np

# 2️⃣ Load
dataset = pd.read_csv('/content/Housing.csv')

# 3️⃣ Prepare
X = dataset.drop('price', axis=1)
y = dataset['price']

le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 4️⃣ Model
model = RandomForestRegressor(n_estimators=50, random_state=0)

# 5️⃣ Train/Test
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
