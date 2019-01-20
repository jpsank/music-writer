from music21 import converter, instrument, note, chord, stream, corpus
import glob
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, LSTM
from keras.callbacks import ModelCheckpoint
import concurrent.futures
import pickle
import os


def one_notes(file):
    add_notes = []

    midi = converter.parse(file)
    # midi.show('text')

    print("Parsing %s" % file)

    parts = midi.parts
    stream_map = {}
    for p in parts:
        for element in p.notes:
            if element.duration.type != 'zero':
                if element.offset not in stream_map:
                    stream_map[element.offset] = []
                instrName = p.getInstrument(returnDefault=True).instrumentName
                item = [element, instrName]
                stream_map[element.offset].append(item)

    last_time = 0
    for time in sorted(stream_map.keys()):
        notes_in_time = [str(time-last_time)]

        for item in stream_map[time]:
            element, instrName = item

            if isinstance(element, note.Note):
                notes_in_time.append("{},{},{}".format(str(element.pitch), element.quarterLength, instrName))
            elif isinstance(element, chord.Chord):
                notes_in_time.append("{},{},{}".format('.'.join(str(n) for n in element.normalOrder), element.quarterLength, instrName))
        add_notes += notes_in_time
        last_time = time

    add_notes.append("$")  # signifies end of sample

    return add_notes


def get_notes():
    """ Get all the notes and chords from the midi files in the midi songs directory """
    all_notes = []

    files = glob.glob("data/midi/*.mid")[:25]

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_idx = {executor.submit(one_notes, file): i
                         for i, file in enumerate(files)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            add = future.result()
            print(idx, add)
            all_notes += add
    # all_notes += one_notes("data/palace.mid")

    with open('data/notes', 'wb') as f:
        pickle.dump(all_notes, f)

    return all_notes


if os.path.exists("data/notes"):
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)
else:
    notes = get_notes()


sequence_length = 100
# get all pitch names
pitchnames = sorted(set(item for item in notes))

n_vocab = len(pitchnames)
print("n_vocab:", n_vocab)

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


filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]
model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

