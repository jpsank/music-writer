import pretty_midi
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


class NeuralNetwork:
    def __init__(self, notes):
        print("Preparing sequences...")
        self.n_vocab, self.network_input, self.network_output, self.note_to_int, self.int_to_note = self.parse_notes(notes)

        print("Creating network...")
        self.network = self.build_network()

    def parse_notes(self, notes, sequence_length=100):
        pitchnames = sorted(set(item for item in notes))

        n_vocab = len(pitchnames)
        print("n_vocab:", n_vocab)

        # create a dictionary to map pitches to integers
        note_to_int = {note: number for number, note in enumerate(pitchnames)}
        int_to_note = {number: note for number, note in enumerate(pitchnames)}

        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        # reshape the input into a format compatible with LSTM layers
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)
        network_output = np_utils.to_categorical(network_output)

        return n_vocab, network_input, network_output, note_to_int, int_to_note

    def build_network(self):
        model = Sequential()

        model.add(LSTM(
            256,
            input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256))
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
        start = 0
        print("start:", start)

        pattern = self.network_input[start]
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


def encode_representation(midi_data):
    add_notes = []

    midi_map = {}
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.start = int(midi_data.time_to_tick(note.start) / 10)
            note.end = int(midi_data.time_to_tick(note.end) / 10)
            if note.start not in midi_map:
                midi_map[note.start] = []
            midi_map[note.start].append(["on", note, instrument])
            if note.end not in midi_map:
                midi_map[note.end] = []
            midi_map[note.end].append(["off", note, instrument])

    last_time = 0
    for time in sorted(midi_map.keys()):
        notes_in_time = [
            "rest{}".format(time - last_time)
        ]
        for item in midi_map[time]:
            on, note, instrument = item
            notes_in_time.append("{}{}-{}".format(on, note.pitch, instrument.program))

        add_notes += notes_in_time
        last_time = time

    if add_notes:
        return ["^"] + add_notes + ["$"]  # signify start and end of sample
    else:
        return []


def decode_representation(rep_sequence, tempo=120):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    timing_map = {}
    instruments = {}
    offset = 0
    for element in rep_sequence:
        if element.startswith("on"):  # note on pointer
            element = element[2:]
            timing_map[element] = offset
        elif element.startswith("off"):  # note off pointer
            element = element[3:]
            pitch, instr_program = element.split("-")
            pitch = int(pitch)
            instr_program = int(instr_program)

            if instr_program not in instruments:
                instr = pretty_midi.Instrument(program=instr_program)
                instruments[instr_program] = instr

            if element in timing_map:  # in case the model creates invalid midi
                new_note = pretty_midi.Note(velocity=100, pitch=pitch, start=timing_map[element], end=offset)
                instruments[instr_program].notes.append(new_note)
        else:
            if element == "$":  # end of song, wait one second
                offset += 1
            elif element.startswith("rest"):  # element must be delay between note/chords
                element = element[4:]
                offset += midi.tick_to_time(int(element) * 10)

    for instr in instruments:
        print(instruments[instr].notes)
        midi.instruments.append(instruments[instr])

    return midi
