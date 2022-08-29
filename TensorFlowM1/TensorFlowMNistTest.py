import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow version:", tf.__version__)
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('List of all physical devices is:')
print(tf.config.list_physical_devices())
print('List of physical CPU devices is:')
print(tf.config.list_physical_devices('CPU'))
print('List of physical GPU devices is:')
print(tf.config.list_physical_devices('GPU'))

def disable_visibility_of_devices_of_type(device_type):
    assert device_type in ['CPU', 'GPU']
    print(f'Disabling all devices of type: {device_type}.')

    # From: https://datascience.stackexchange.com/questions/58845/how-to-disable-gpu-with-tensorflow
    # answer of tttzof351
    try:
        tf.config.set_visible_devices([], device_type)

        # check if it worked:
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != device_type
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

disable_cpu_visibility = False  # <-- You cannot choose, must be False
assert not disable_cpu_visibility
if disable_cpu_visibility:
    disable_visibility_of_devices_of_type('CPU')  # Gives: No CPU devices are available in this process, so avoid call.

disable_gpu_visibility = False  # <-- You can choose, False is fastest.
if disable_gpu_visibility:
    disable_visibility_of_devices_of_type('GPU')

# eagerly: For its meaning, see: https://www.tensorflow.org/guide/intro_to_graphs
eagerly = False  # <-- You can choose, False is fastest.

print('---')
print(f'disable_cpu_visibility = {disable_cpu_visibility}')
print(f'disable_gpu_visibility = {disable_gpu_visibility}')
print(f'eagerly = {eagerly}')
print('---')

print('List of visible devices is:')
print(tf.config.get_visible_devices())

tf.config.run_functions_eagerly(eagerly) # Only seems to have effect when restarting the jupyter notebook.
print(f'tf.executing_eagerly() = {tf.executing_eagerly()}')  # Gives True when line above set it to False!

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

tf.config.list_physical_devices('GPU')
from timeit import default_timer as timer
from datetime import timedelta

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

