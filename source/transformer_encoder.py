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

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        # Define the 1x1 convolutions for ACmix
        self.qkv_conv = tf.keras.layers.Conv2D(filters=self.d_model * 3, kernel_size=(1, 1), strides=(1, 1))

        # Fully connected layer for ACmix
        self.fc = tf.keras.layers.Conv2D(filters=self.d_model, kernel_size=(1, 1), strides=(1, 1))

        # Depthwise convolution for ACmix
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')

        # Learnable parameters for combining the self-attention and convolution outputs
        self.alpha = self.add_weight(name='alpha', shape=[], initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=[], initializer='ones', trainable=True)

        self.poswiseff_layer = PosWiseFF(self.d_model, self.d_ff, self.atv_fun, self.dropout_rate, name='pos_wise_ff')
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='enc_norm1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='enc_norm2')

    def call(self, inputs, mask=None):
        x = inputs
        B, L, E = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Convert L to a float, apply sqrt, then convert back to an integer
        sqrt_L = tf.cast(tf.sqrt(tf.cast(L, tf.float32)), tf.int32)

        # Ensure that the sqrt_L * sqrt_L is not greater than L
        sqrt_L = tf.minimum(sqrt_L, tf.cast(tf.sqrt(tf.cast(L, tf.float32)), tf.int32))

        # Reshape input for convolution operations
        x_reshaped = tf.reshape(x, (B, sqrt_L, sqrt_L, E))

        # Apply 1x1 convolution (qkv_conv)
        qkv = self.qkv_conv(x_reshaped)
        q, k, v = tf.split(qkv, 3, axis=-1)

        # Self-attention operation
        attn_out = self.self_attention(q, k, v, mask)

        # Apply fully connected layer
        f_out = self.fc(qkv)

        # Apply depthwise convolution
        conv_out = self.depthwise_conv(f_out)

        # Reshape the output back to the original shape
        conv_out = tf.reshape(conv_out, (B, L, E))

        # Combine self-attention and convolution outputs
        attn_out = self.alpha * attn_out + self.beta * conv_out

        sublayer1_out = self.layernorm1(x + attn_out)

        # Position-Wise Feed Forward
        poswiseff_out = self.poswiseff_layer(sublayer1_out)
        sublayer2_out = self.layernorm2(sublayer1_out + poswiseff_out)

        return sublayer2_out

    def self_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (B, H*W, H*W)

        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply the mask (if provided)
        if mask is not None:
            # Reshape the mask to make it broadcastable to the shape of scaled_attention_logits
            mask = tf.reshape(mask, [tf.shape(mask)[0], 1, 1, tf.shape(mask)[-1]])
            scaled_attention_logits += (mask * -1e9)

        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (B, H*W, H*W)

        output = tf.matmul(attention_weights, v)  # (B, H*W, C)

        return output


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
