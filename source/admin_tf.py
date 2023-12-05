import tensorflow as tf
import math

class Admin(tf.keras.layers.Layer):
    def __init__(self, num_res_layers, **kwargs):
        super(Admin, self).__init__(**kwargs)

        # self.omega_value = ((num_res_layers + 1) / math.log(num_res_layers + 1) - 1) ** .5
    
    def build(self, input_shape):
        self.omega = self.add_weight(name="admin-parameter",
                                     trainable=False,
                                     shape=input_shape[-1],
                                     initializer=tf.constant_initializer(1))

    def call(self, x, f_x):
        return x * self.omega + f_x
    
    def get_config(self):
        config = super(Admin, self).get_config()
        config.update({
            'omega_value' : self.omega_value 
        })
