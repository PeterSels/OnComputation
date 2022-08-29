import tensorflow as tf
import tensorflow_datasets as tfds

from timeit import default_timer as timer
from datetime import timedelta

from EagerCpuGpuConfig import EagerCpuGpuConfig

cfg = EagerCpuGpuConfig(eagerly=False, disable_cpu_visibility=False, disable_gpu_visibility=False)
cfg.print_parameters()
cfg.print_results()

"""
I get on a Apple MacBook Pro 16" M1 Max, for the model fitting (training) the process times:
https://henk-celcius.medium.com/installing-and-cpu-vs-gpu-testing-tensorflow-on-an-apple-mac-m1-arm-native-via-miniforge-90d49eaf05ea
MacM1MaxEagerCpuNoGpu.log:          0:05:13.037244
MacM1MaxNotEagerCpuNoGpu.log:       0:03:58.723621
MacM1MaxEagerCpuGpu.log:            0:01:33.897871
MacM1MaxNotEagerCpuGpu.log:         0:00:51.107395

I get on a Dell Precision i7 9850H CPU @ 2.6GHz and 16 GB RAM with Intel UHD Graphics 630 GPU
https://henk-celcius.medium.com/installing-tensorflow-on-a-wintel-machine-with-quadro-t1000-gpu-performance-comparison-with-and-c64b6c1b9cd1

WinQuadroT1000EagerCpuNoGpu.log:    0:11:05.841665
WinQuadroT1000NotEagerCpuNoGpu.log: 0:07:03.279670
Wini7QuadroT1000EagerCpuGpu.log:    0:02:09.316872
Wini7QuadroT1000NotEagerCpuGpu.log: 0:01:22.671263
"""

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

batch_size = 128

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu'),
  tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                 activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#   tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

start = timer()
model.fit(
    ds_train,
    epochs=12,
    validation_data=ds_test,
)
end = timer()
print(timedelta(seconds=end-start))

