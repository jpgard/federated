import tensorflow as tf


# create a simple keras model
def create_compiled_keras_model(input_shape):
    """Create and compile a simple fully-connected Keras model."""
    model = tf.keras.models.Sequential([
        # preprocessing_layer,
        tf.keras.layers.BatchNormalization(center=True, scale=True,
                                           input_shape=input_shape),
        tf.keras.layers.Dense(10, activation='relu', kernel_initializer='zeros'),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='zeros')
    ])
    # compile the model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        metrics=[
            tf.keras.metrics.BinaryCrossentropy(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ])
    return model
