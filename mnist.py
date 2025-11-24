# 1️⃣ Import
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 2️⃣ Load
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 3️⃣ Prepare (reshape + normalize)
x_train = x_train.reshape(60000, 28*28).astype('float32') / 255
x_test  = x_test.reshape(10000, 28*28).astype('float32') / 255

# 4️⃣ Model (build the neural network)
model = Sequential()
model.add(Dense(32, input_dim=28*28, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 digits (0–9)

# 5️⃣ Train + Evaluate
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128)
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# Optional: Predict one sample
y_pred = model.predict(x_test)
print("Predicted digit:", np.argmax(y_pred[0]))



# from keras.datasets import mnist
# data=mnist.load_data()
# print(data)

# (x_train, y_train), (x_test, y_test)= data
# x_train [0].shape
# x_train.shape

# x_test_copy = x_test [:] [:]
# y_train [0]

# import matplotlib.pyplot as plt
# plt.imshow(x_test [1], cmap ='gray')

# y_train [0]

# x_train= x_train.reshape ((x_train. shape[0], 28*28)).astype('float32')
# x_test=x_test.reshape ((x_test.shape[0], 28*28)).astype('float32')
# x_train= x_train / 255
# x_test =x_test / 255
# x_train.shape

# from keras.models import Sequential
# from keras.layers import Dense
# model=Sequential ()
# model.add(Dense(32,input_dim= 28*28, activation = 'relu'))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(32, activation = 'softmax'))

# model.compile(loss= 'sparse_categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])

# model.fit (x_train, y_train, epochs=20, batch_size=128)

# loss, acc = model.evaluate (x_test, y_test, batch_size=128)

# y_pred=model.predict(x_test)

# plt.imshow(x_test_copy [0], cmap="gray")

# y_pred [0]
# import numpy as np
# np. argmax(y_pred[0], axis = -1)
# y_test[0]

