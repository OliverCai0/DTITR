import tensorflow as tf
import math

class Admin(tf.keras.layers.Layer):
    def __init__(self, num_res_layers, input_shape, **kwargs):
        super(Admin, self).__init__(**kwargs)

        self.omega_value = ((num_res_layers + 1) / math.log(num_res_layers + 1) - 1) ** .5
    
    def build(self, input_shape):
        self.omega = self.add_weight(trainable=True,
                                     initializer=tf.constant_initializer(self.omega_value))

    def call(self, x, f_x):
        return x * self.omega + f_x
