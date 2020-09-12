from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# # config.log_device_placement = True  # to log device placement (on which device the operation ran)
#
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

SEQUENCE_LENGTH = 100


class NeuralNetwork:
    def __init__(self, notes, sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length

        print("Preparing sequences...")
        self.n_vocab, self.network_input, self.network_output, self.note_to_int, self.int_to_note = \
            self.parse_notes(notes)

        self.primer = None

        print("Creating network...")
        self.network = self.build_network()

    def parse_notes(self, notes):
        pitchnames = sorted(set(item for item in notes))

        n_vocab = len(pitchnames)
        print("n_vocab:", n_vocab)

        # create a dictionary to map pitches to integers
        note_to_int = {note: number for number, note in enumerate(pitchnames)}
        int_to_note = {number: note for number, note in enumerate(pitchnames)}

        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - self.sequence_length, 1):
            sequence_in = notes[i:i + self.sequence_length]
            sequence_out = notes[i + self.sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        # reshape the input into a format compatible with LSTM layers
        network_input = np.reshape(network_input, (n_patterns, self.sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)
        network_output = np_utils.to_categorical(network_output)

        return n_vocab, network_input, network_output, note_to_int, int_to_note

    def set_primer(self, notes):
        self.primer = notes[:self.sequence_length]

    def build_network(self):
        model = Sequential()

        model.add(LSTM(
            512,
            input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        return model

    def load_weights(self, filepath):
        self.network.load_weights(filepath)

    def train(self):
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]
        self.network.fit(self.network_input, self.network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

    def predict(self, n=500):
        if self.primer:
            pattern = self.primer
        else:
            pattern = self.network_input[0]
        prediction_output = []

        # generate n notes
        for note_index in range(n):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(self.n_vocab)

            prediction = self.network.predict(prediction_input, verbose=0)

            index = np.argmax(prediction)
            result = self.int_to_note[index]
            prediction_output.append(result)

            if result == "$":
                break

            pattern = np.vstack((pattern, index))
            pattern = pattern[1:len(pattern)]

        return prediction_output
