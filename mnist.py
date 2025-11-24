# 1️⃣ Import
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 2️⃣ Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Make a copy so we can show images later (not flattened)
x_test_copy = x_test.copy()

# 3️⃣ Show a test image (index 1)
index = 1     # YOU CAN CHANGE THIS TO ANY DIGIT YOU WANT TO TEST
plt.imshow(x_test_copy[index], cmap='gray')
plt.title(f"Actual Digit: {y_test[index]}")
plt.show()

# 4️⃣ Prepare the data (flatten + normalize)
x_train = x_train.reshape(60000, 28*28).astype('float32') / 255
x_test  = x_test.reshape(10000, 28*28).astype('float32') / 255

# 5️⃣ Build the model
model = Sequential()
model.add(Dense(32, input_dim=28*28, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))   # 10 classes (0–9)

# 6️⃣ Train
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=128)

# 7️⃣ Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# 8️⃣ Predict the SAME image shown above
y_pred = model.predict(x_test)
predicted_digit = np.argmax(y_pred[index])

print("Predicted Digit for shown image:", predicted_digit)
print("Actual Digit:", y_test[index])

# 9️⃣ Plot training accuracy
plt.plot(history.history['accuracy'])
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# 🔟 Plot training loss
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
