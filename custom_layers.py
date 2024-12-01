import tensorflow as tf
from keras.layers import LayerNormalization, Dense

class CWTLayer(tf.keras.layers.Layer):
    def __init__(self, num_scales, scale_factor=1.0, wavelet='cmor', **kwargs):
        super(CWTLayer, self).__init__(**kwargs)
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.wavelet = wavelet

    def build(self, input_shape):
        super(CWTLayer, self).build(input_shape)

    def call(self, inputs):
        scales = tf.range(1, self.num_scales + 1, dtype=tf.float32)
        scales = scales * self.scale_factor

        # Create a 2D tensor with shape (num_scales, 1) for broadcasting
        scales = tf.reshape(scales, (1, -1, 1))

        # Perform CWT
        cwt_result = tf.signal.cwt(tf.squeeze(inputs, axis=-1), scales, self.wavelet)

        # Expand dimensions to match the input shape
        cwt_result = tf.expand_dims(cwt_result, axis=-1)

        # Add skip connection.
        x = cwt_result + inputs

        # Apply layer normalization.
        x = LayerNormalization(epsilon=1e-6)(x)

        # Apply Feedforward network.
        x_ffn = Dense(units=256, activation="gelu")(x)

        # Add skip connection.
        x = x + x_ffn

        # Apply layer normalization.
        x = LayerNormalization(epsilon=1e-6)(x)

        return x


class FNetLayer(tf.keras.layers.Layer):
    def __init__(self, units=256, activation="gelu", **kwargs):
        super(FNetLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        super(FNetLayer, self).build(input_shape)

    def call(self, inputs):
        x = tf.cast(
            tf.signal.fft(tf.cast(inputs, dtype=tf.dtypes.complex64)),
            dtype=tf.dtypes.float32,
        )
        # Add skip connection.
        x = x + inputs
        # Apply layer normalization.
        x = LayerNormalization(epsilon=1e-6)(x)
        # Apply Feedforward network.
        x_ffn = Dense(units=self.units, activation=self.activation)(x)
        # Add skip connection.
        x = x + x_ffn
        # Apply layer normalization.
        x = LayerNormalization(epsilon=1e-6)(x)
        return x
