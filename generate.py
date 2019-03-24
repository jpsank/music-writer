import pretty_midi
import glob
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, LSTM
from keras.callbacks import ModelCheckpoint
import concurrent.futures
import pickle, os


with open('data/notes', 'rb') as filepath:
    notes = pickle.load(filepath)

# print(notes)
print("Preparing sequences...")

sequence_length = 100
# get all pitch names
pitchnames = sorted(set(item for item in notes))

n_vocab = len(pitchnames)
print("n_vocab: {}".format(n_vocab))

# create a dictionary to map pitches to integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
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


print("Creating network...")

model = Sequential()
model.add(LSTM(
    256,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Load the weights to each node
model.load_weights('weights-improvement-10-2.8671-bigger.hdf5')

print("Generating notes...")

start = np.random.randint(0, len(network_input)-1)
print("start:", start)

int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

pattern = network_input[start]
prediction_output = []

# generate 500 notes
for note_index in range(300):  # default 500
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)

    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)

    pattern = np.vstack((pattern,index))
    pattern = pattern[1:len(pattern)]

print("Prediction output:", prediction_output)
print("Creating midi...")

midi = pretty_midi.PrettyMIDI()


timing_map = {}
instruments = {}
offset = 0
for element in prediction_output:
    if element.startswith("on"):  # note on pointer
        element = element[2:]
        timing_map[element] = offset
    elif element.startswith("off"):  # note off pointer
        element = element[3:]
        pitch, instrProgram = element.split("-")
        pitch = int(pitch)
        instrProgram = int(instrProgram)

        if instrProgram not in instruments:
            instr = pretty_midi.Instrument(program=instrProgram)
            instruments[instrProgram] = instr

        new_note = pretty_midi.Note(velocity=100, pitch=pitch, start=timing_map[element], end=offset)
        instruments[instrProgram].notes.append(new_note)
    else:
        if element == "$":  # end of song, wait one second
            offset += 1
        else:  # element must be delay between note/chords
            offset += float(element)

for instr in instruments:
    midi.instruments.append(instruments[instr])

midi.write('test_output.mid')

print("Done")
