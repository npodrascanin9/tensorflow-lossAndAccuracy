import numpy as np

train_images = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
train_labels = np.array([[1], [0], [0], [0]])

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(4, input_shape = (2,), activation = "relu"),
    layers.Dense(1, activation = "sigmoid")
])

from tensorflow.keras import optimizers, losses, metrics

model.compile(optimizer=optimizers.Adam(learning_rate=0.1),
                loss=losses.mean_squared_error,
                metrics=[metrics.BinaryAccuracy()])

history = model.fit(train_images, train_labels, epochs = 50)

print(history.history.keys())

import matplotlib.pylab as plt

loss_values = history.history["loss"]
accuracy_values = history.history["binary_accuracy"]
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'r--')
plt.plot(epochs, loss_values, 'ro', label = 'Training loss')
plt.plot(epochs, accuracy_values, 'b', label = 'Tracking accuracy')
plt.title('Training loss and training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss, Accuracy')
plt.legend()
plt.show()

predictions = model.predict(train_images)
print(predictions)
