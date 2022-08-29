import tensorflow as tf
import tensorflow_datasets as tfds

from timeit import default_timer as timer
from datetime import timedelta

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.datasets.cifar10 import load_data

import matplotlib.pyplot as plt
import numpy as np

from EagerCpuGpuConfig import EagerCpuGpuConfig

cfg = EagerCpuGpuConfig(eagerly=False, disable_cpu_visibility=False, disable_gpu_visibility=False)
cfg.print_parameters()
cfg.print_results()

(X_train, y_train), (X_test, y_test) = load_data()

# rescale image
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

model = Sequential([
    Conv2D(32, (3,3), input_shape=(32, 32, 3), padding="same", activation="relu", kernel_constraint=MaxNorm(3)),
    Dropout(0.3),
    Conv2D(32, (3,3), padding="same", activation="relu", kernel_constraint=MaxNorm(3)),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation="relu", kernel_constraint=MaxNorm(3)),
    Dropout(0.5),
    Dense(10, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", 
              metrics="sparse_categorical_accuracy")

model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=25, batch_size=32)

model.summary()

print(model.layers[0].kernel)

# Extract output from each layer
extractor = tf.keras.Model(inputs=model.inputs,
                           outputs=[layer.output for layer in model.layers])
features = extractor(np.expand_dims(X_train[7], 0))

# Show the 32 feature maps from the first layer
l0_features = features[0].numpy()[0]

fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16, 8))
for i in range(0, 32):
    row, col = i // 8, i % 8
    ax[row][col].imshow(l0_features[..., i])

plt.show()

# Show the 32 feature maps from the third layer
l2_features = features[2].numpy()[0]

fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16, 8))
for i in range(0, 32):
    row, col = i // 8, i % 8
    ax[row][col].imshow(l2_features[..., i])

plt.show()


