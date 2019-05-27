import pretty_midi
import glob
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, LSTM
from keras.callbacks import ModelCheckpoint
import concurrent.futures
import os
import json


def one_notes(file):
    add_notes = []

    print("Parsing %s" % file)

    midi_data = pretty_midi.PrettyMIDI(file)

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
            "rest{}".format(time-last_time)
        ]
        for item in midi_map[time]:
            on, note, instrument = item
            notes_in_time.append("{}{}-{}".format(on, note.pitch, instrument.program))

        add_notes += notes_in_time
        last_time = time

    if add_notes:
        add_notes.append("$")  # signifies end of sample

        return add_notes
    else:
        return []


def get_notes():
    """ Get all the notes and chords from the midi files in the midi songs directory """
    all_notes = []

    # files = glob.glob("data/midi/*.mid")[:2]
    files = ["data/midi/Fugue1.mid"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_idx = {executor.submit(one_notes, file): i
                         for i, file in enumerate(files)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            add = future.result()
            print("{}/{}".format(idx+1, len(files)), add)
            all_notes += add
    # all_notes += one_notes("data/midi/palace.mid")

    with open('data/notes', 'wb') as f:
        json.dump(all_notes, f)

    return all_notes


if os.path.exists("data/notes"):
    with open('data/notes', 'rb') as f:
        notes = json.load(f)
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

