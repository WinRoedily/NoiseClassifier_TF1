import os

os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

import tensorflow as tf
import h5py

num_cep = 40
num_frames = 102

hf = h5py.File('hdf5\\features_train_{0}.h5'.format(num_frames), 'r')
x_train = hf.get('data')[()]
y_train = hf.get('label')[()]

hf2 = h5py.File('hdf5\\features_test_{0}.h5'.format(num_frames), 'r')
x_test = hf2.get('data')[()]
y_test = hf2.get('label')[()]

y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)

epochs = 100


def lstm_layer(inputs):
    """
    Build LSTM Layer

    :param inputs: input tensor
    :param num_layers: number of LSTM layer
    :param num_units: number of LSTM unit in the cell
    :return: unstacked output
    """
    lstm_layers = tf.keras.layers.StackedRNNCells([
        tf.lite.experimental.nn.TFLiteLSTMCell(64, forget_bias=1.0, initializer=tf.keras.initializers.glorot_uniform),
        tf.lite.experimental.nn.TFLiteLSTMCell(32, forget_bias=1.0, initializer=tf.keras.initializers.glorot_uniform)])

    # Transpose input [batch, time, input_size]
    transpose = tf.transpose(inputs, perm=[1, 0, 2])
    outputs, _ = tf.lite.experimental.nn.dynamic_rnn(
        lstm_layers, transpose, dtype='float32', time_major=True)
    unstacked_output = tf.unstack(outputs, axis=0)

    return unstacked_output[-1]


tf.reset_default_graph()
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(num_frames, num_cep), name='input'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu',
                           kernel_initializer=tf.keras.initializers.he_uniform()),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='valid', activation='relu',
                           kernel_initializer=tf.keras.initializers.he_uniform()),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Lambda(lstm_layer),
    tf.keras.layers.Dense(5, activation='softmax', name='output')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

sess = tf.keras.backend.get_session()
input_tensor = sess.graph.get_tensor_by_name('input:0')
output_tensor = sess.graph.get_tensor_by_name('output/Softmax:0')
converter = tf.lite.TFLiteConverter.from_session(
    sess, [input_tensor], [output_tensor])
converter.experimental_new_converter = True
tflite = converter.convert()
open(file="classifier_{0}.tflite".format(num_frames), mode="wb").write(tflite)
print("Convert DONE")
