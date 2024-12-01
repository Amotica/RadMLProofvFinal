# Load libraries
from keras.layers import *
from keras.models import Model
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import Precision, Recall


def bs_sparse_categorical_crossentropy(y_true, y_pred):
    # Calculate sparse categorical crossentropy
    sparse_categorical_crossentropy = SparseCategoricalCrossentropy()(y_true, y_pred)

    # Compute the number of elements in each batch
    batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float64)

    # Compute the number of bootstrap samples
    n_bs_samples = tf.cast(tf.shape(y_true)[1], dtype=tf.float64)

    # Cast y_true to float64
    y_true_float64 = tf.cast(y_true, dtype=tf.float64)

    # Calculate the weight for each sample
    sample_weights = tf.reduce_sum(y_true_float64, axis=-1) / (batch_size * n_bs_samples)

    # Explicitly cast both tensors to float64 before multiplication
    sample_weights_float64 = tf.cast(sample_weights, dtype=tf.float64)
    sparse_categorical_crossentropy_float64 = tf.cast(sparse_categorical_crossentropy, dtype=tf.float64)

    # Apply the bootstrapped weights to the sparse categorical crossentropy
    bs_loss = tf.reduce_sum(sample_weights_float64 * sparse_categorical_crossentropy_float64)

    return bs_loss


def SNetLayer(inputs, frame_length=256, frame_step=2):
    x = tf.cast(
        tf.signal.stft(
            signals=inputs,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=frame_length
        ),
        dtype=tf.float32
    )
    x = Dense(units=256)(x)
    x = tf.reshape(x, shape=(-1, 256))
    # Add skip connection.
    x = x + inputs
    # Apply layer normalization.
    x = LayerNormalization(epsilon=1e-6)(x)
    # Apply Feedforward network.
    x_ffn = Dense(units=256, activation="gelu")(x)
    # Add skip connection.
    x = x + x_ffn
    # Apply layer normalization.
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def FNetLayer(inputs):
    x = tf.cast(
        tf.signal.fft(tf.cast(inputs, dtype=tf.dtypes.complex64)),
        dtype=tf.dtypes.float32,
    )
    # Add skip connection.
    x = x + inputs
    # Apply layer normalization.
    x = LayerNormalization(epsilon=1e-6)(x)
    # Apply Feedforward network.
    x_ffn = Dense(units=256, activation="gelu")(x)
    # Add skip connection.
    x = x + x_ffn
    # Apply layer normalization.
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def BasicFNetLayer(inputs):
    x = tf.cast(
        tf.signal.fft(tf.cast(inputs, dtype=tf.dtypes.complex64)),
        dtype=tf.dtypes.float32
    )
    x = LayerNormalization(epsilon=1e-6)(x)
    # Apply Feedforward network.
    x_ffn = Dense(units=256, activation="gelu")(x)
    # Add skip connection.
    x = x + x_ffn
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def SNetModelDouble(features, num_classes, testing=False):
    inputs = Input(shape=(features,))
    x = Dense(units=256)(inputs)

    # Could repeat this block many times. Currently, only one layer
    x = SNetLayer(x)
    x = SNetLayer(x)

    x = Dense(num_classes, activation='softmax')(x)
    # Create the Keras model.
    model = Model(inputs=inputs, outputs=x, name="SNetModelDouble")
    if testing:
        model.compile(loss=bs_sparse_categorical_crossentropy, optimizer="adam", metrics = ['accuracy', Precision(), Recall()])
    else:
        model.compile(loss=bs_sparse_categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
    return model


def FNetModelDouble(features, num_classes, testing=False):
    inputs = Input(shape=(features,))
    x = Dense(units=256)(inputs)

    # Could repeat this block many times. Currently, only one layer
    x = FNetLayer(x)
    x = FNetLayer(x)

    x = Dense(num_classes, activation='softmax')(x)
    # Create the Keras model.
    model = Model(inputs=inputs, outputs=x, name="FNetModelDouble")
    if testing:
        model.compile(loss=bs_sparse_categorical_crossentropy, optimizer="adam",
                      metrics=['accuracy', Precision(), Recall()])
    else:
        model.compile(loss=bs_sparse_categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
    return model


def BasicFNetModel(features, num_classes, testing=False):
    inputs = Input(shape=(features,))
    x = Dense(units=256)(inputs)

    # Could repeat this block many times. Currently, only one layer
    x = BasicFNetLayer(x)
    x = BasicFNetLayer(x)

    x = Dense(num_classes, activation='softmax')(x)
    # Create the Keras model.
    model = Model(inputs=inputs, outputs=x, name="BasicFNetModel")
    if testing:
        model.compile(loss=bs_sparse_categorical_crossentropy, optimizer="adam",
                      metrics=['accuracy', Precision(), Recall()])
    else:
        model.compile(loss=bs_sparse_categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

    return model