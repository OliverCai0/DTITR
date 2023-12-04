# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

from lmha_layer import *
from layers_utils import *
from mha_layer import *


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder Layer of the Transformer-Encoder

    Args:
    - d_model [int]: embedding dimension
    - num_heads [int]: number of heads of attention
    - d_ff [int]: number of hidden neurons for the first dense layer of the FFN
    - atv_fun: dense layers activation function
    - dropout_rate [float]: % of dropout
    - dim_k [int]: Linear MHA projection dimension
    - parameter_sharing [str]: Linear MHA parameter sharing option
    - full_attention [boolean]: True - original O(n2) attention, False - Linear Attention

    """

    def __init__(self, d_model, num_heads, d_ff, atv_fun, dropout_rate, dim_k, parameter_sharing, full_attention,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate
        self.dim_k = dim_k
        self.parameter_sharing = parameter_sharing
        self.full_attention = full_attention

        # Define convolutional layers for the convolution path
        self.conv_q = tf.keras.layers.Conv2D(filters=d_model, kernel_size=1, use_bias=True)
        self.conv_k = tf.keras.layers.Conv2D(filters=d_model, kernel_size=1, use_bias=True)
        self.conv_v = tf.keras.layers.Conv2D(filters=d_model, kernel_size=1, use_bias=True)

        # ACMix: Initialize alpha and beta
        self.alpha = tf.Variable(initial_value=1.0, trainable=True)
        self.beta = tf.Variable(initial_value=1.0, trainable=True)

    def build(self, input_shape):

        if self.full_attention:
            self.mha_layer = MultiHeadAttention(self.d_model, self.num_heads, self.dropout_rate,
                                                name='enc_self_attn')

        else:
            self.E_proj = linear_proj_matrix(self.dim_k)

            self.mha_layer = LMHAttention(self.d_model, self.num_heads, self.dropout_rate, self.parameter_sharing,
                                          self.dim_k,
                                          self.E_proj, self.E_proj, name='enc_self_lattn')

        self.poswiseff_layer = PosWiseFF(self.d_model, self.d_ff, self.atv_fun, self.dropout_rate, name='pos_wise_ff')

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='enc_norm1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='enc_norm2')
    
    def convolution_path(self, q_proj, k_proj, v_proj):
        # Reshape projected features for Conv2D compatibility
        q_proj_reshaped = tf.expand_dims(q_proj, axis=2)
        k_proj_reshaped = tf.expand_dims(k_proj, axis=2)
        v_proj_reshaped = tf.expand_dims(v_proj, axis=2)

        # Apply predefined convolutional layers
        conv_q = self.conv_q(q_proj_reshaped)
        conv_k = self.conv_k(k_proj_reshaped)
        conv_v = self.conv_v(v_proj_reshaped)

        # Combine convolution outputs
        combined_conv_output = conv_q + conv_k + conv_v

        # Reshape back to original dimensions (if necessary)
        combined_conv_output = tf.squeeze(combined_conv_output, axis=2)

        return combined_conv_output
    
    def call(self, inputs, mask=None):
        """

        Args:
        - inputs: input sequences
        - mask: attention weights mask

        Shape:
        - Inputs:
        - inputs: (B,L,E): where B is the batch size, L is the input sequence length,
                        E is the embedding dimension
        - mask: (B,1,1,L): where B is the batch size, L is the input sequence length

        - Outputs:
        - sublayer2_out: (B,E,L):  where B is the batch size, L is the input sequence length,
                        E is the embedding dimension
        - attn_w: (B,H,L,L): where B is the batch size, H is the number of heads,
                        L is the input sequence length.

        """

        # Reshape the input to 4D for compatibility with Conv2D
        # Adding a singleton 'width' dimension
        inputs_reshaped = tf.expand_dims(inputs, axis=2)  # New shape: (batch_size, 557, 1, 128)

        # Use the correct convolutional layers
        q_proj = self.conv_q(inputs_reshaped)
        k_proj = self.conv_k(inputs_reshaped)
        v_proj = self.conv_v(inputs_reshaped)

        # Reshape the convoluted outputs back to the original dimensions
        # Removing the singleton dimension added earlier
        q_proj = tf.squeeze(q_proj, axis=2)
        k_proj = tf.squeeze(k_proj, axis=2)
        v_proj = tf.squeeze(v_proj, axis=2)

        # Self-Attention Path
        attn_out, attn_w = self.mha_layer([q_proj, k_proj, v_proj], mask=mask)

        # Convolution Path
        conv_out = self.convolution_path(q_proj, k_proj, v_proj)

        # Combine Outputs
        combined_output = self.alpha * attn_out + self.beta * conv_out

        # Continue with existing layer normalization and FFN
        sublayer1_out = self.layernorm1(inputs + combined_output)
        poswiseff_out = self.poswiseff_layer(sublayer1_out)
        sublayer2_out = self.layernorm2(sublayer1_out + poswiseff_out)

        return sublayer2_out, attn_w

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'atv_fun': self.atv_fun,
            'dropout_rate': self.dropout_rate,
            'dim_k': self.dim_k,
            'parameter_sharing': self.parameter_sharing,
            'full_attention': self.full_attention})

        return config


class Encoder(tf.keras.Model):
    """
    Transformer-Encoder

    Args:
    - d_model [int]: embedding dimension
    - num_layers [int]: number of transformer-encoder layers
    - num_heads [int]: number of heads of attention
    - d_ff [int]: number of hidden neurons for the first dense layer of the FFN
    - atv_fun: dense layers activation function
    - dropout_rate [float]: % of dropout
    - dim_k [int]: Linear MHA projection dimension
    - parameter_sharing [str]: Linear MHA parameter sharing option
    - full_attention [boolean]: full_attention [boolean]: True - original O(n2) attention, False - Linear Attention
    - return_intermediate [boolean]: True - returns the intermediate results

    """
    def __init__(self, d_model, num_layers, num_heads, d_ff, atv_fun, dropout_rate,
                 dim_k, parameter_sharing, full_attention, return_intermediate=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate
        self.dim_k = dim_k
        self.parameter_sharing = parameter_sharing
        self.full_attention = full_attention
        self.return_intermediate = return_intermediate

    def build(self, input_shape):
        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.atv_fun,
                                        self.dropout_rate, self.dim_k, self.parameter_sharing,
                                        self.full_attention, name='layer_enc%d' % i)
                           for i in range(self.num_layers)]

    def call(self, inputs, mask=None):
        """

        Args:
        - inputs: input sequences
        - mask: attention weights mask

        Shape:
        - Inputs:
        - inputs: (B,L,E): where B is the batch size, L is the input sequence length,
                        E is the embedding dimension
        - mask: (B,1,1,L): where B is the batch size, L is the input sequence length

        - Outputs:
        - x: (B,E,L):  where B is the batch size, L is the input sequence length,
                        E is the embedding dimension
        - attention_weights: dictionary with the attentions weights (B,H,L,L) of each encoder layer

        """

        x = inputs
        intermediate = []
        attention_weights = {}

        for layer in self.enc_layers:
            x, attn_enc_w = layer(x, mask)

            if self.return_intermediate:
                intermediate.append(x)

            attention_weights['encoder_layer{}'.format(self.enc_layers.index(layer) + 1)] = attn_enc_w

        if self.return_intermediate:
            return tf.stack(intermediate, axis=0), attention_weights

        return x, attention_weights

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'atv_fun': self.atv_fun,
            'dropout_rate': self.dropout_rate,
            'dim_k': self.dim_k,
            'parameter_sharing': self.parameter_sharing,
            'full_attention': self.full_attention,
            'return_intermediate': self.return_intermediate,
            'enc_layers': self.enc_layers})

        return config
