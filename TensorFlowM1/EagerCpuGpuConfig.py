import tensorflow as tf

class EagerCpuGpuConfig():
    def __init__(self, eagerly=False, disable_cpu_visibility=False, disable_gpu_visibility=False):
        self.eagerly = eagerly  # <-- You can choose, False is fastest.
        #  For its meaning, see: https://www.tensorflow.org/guide/intro_to_graphs
        self.disable_cpu_visibility = disable_cpu_visibility  # <-- You cannot choose, must be False
        self.disable_gpu_visibility = disable_gpu_visibility  # <-- You can choose, False is fastest.

        assert not disable_cpu_visibility
        if disable_cpu_visibility:
            self.disable_visibility_of_devices_of_type(
                'CPU')  # Gives: No CPU devices are available in this process, so avoid call.

        if disable_gpu_visibility:
            self.disable_visibility_of_devices_of_type('GPU')

        tf.config.run_functions_eagerly(eagerly)

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

    def print_parameters(self):
        print('---')
        print(f'disable_cpu_visibility = {self.disable_cpu_visibility}')
        print(f'disable_gpu_visibility = {self.disable_gpu_visibility}')
        print(f'eagerly = {self.eagerly}')
        print('---')

    def print_results(self):
        self.print_physical_devices()
        self.print_eagerly()
        self.print_visible_devices()

    def print_physical_devices(self):
        print("TensorFlow version:", tf.__version__)
        print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print('List of all physical devices is:')
        print(tf.config.list_physical_devices())
        print('List of physical CPU devices is:')
        print(tf.config.list_physical_devices('CPU'))
        print('List of physical GPU devices is:')
        print(tf.config.list_physical_devices('GPU'))

    def print_eagerly(self):
        print(f'tf.executing_eagerly() = {tf.executing_eagerly()}')  # Gives True when line above set it to False!

    def print_visible_devices(self):
        print('List of visible devices is:')
        print(tf.config.get_visible_devices())

