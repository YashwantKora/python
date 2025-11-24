# 1️⃣ Import
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

# 2️⃣ Load
df = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\hhhhhh\python\datasets\and.csv")

# 3️⃣ Prepare
X = df.iloc[:, :2].values
Y = df.iloc[:, 2:].values

# 4️⃣ Model
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 5️⃣ Train
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
model.fit(X, Y, epochs=250)

# 🔍 Optional Testing (not required for exam)
print(model.predict(np.array([[1, 1]])))  # should give value near 1
weights = model.get_weights()
print(weights[1])
