import pretty_midi
import glob
import numpy as np
import concurrent.futures
import os
import json

from model import encode_representation, NeuralNetwork


def sample_to_notes(file):
    print("Parsing %s" % file)

    try:
        midi_data = pretty_midi.PrettyMIDI(file)
        return encode_representation(midi_data)
    except Exception as e:
        print("ERROR", e)
        return []


def get_notes():
    """ Get all the notes and chords from the midi files in the midi songs directory """
    all_notes = []

    # files = glob.glob("data/midi/beethoven/*.mid")
    # files = ["data/midi/beethoven/firstmvm.mid","data/midi/beethoven/secondmv.mid","data/midi/beethoven/thirdmvm.mid"]
    files = ["data/midi/Pokemon XY - Battle Wild Pokemon.mid"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_idx = {executor.submit(sample_to_notes, file): i
                         for i, file in enumerate(files)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            add = future.result()
            print("{}/{}".format(idx+1, len(files)), add)
            all_notes += add

    with open('data/notes.json', 'w') as f:
        json.dump(all_notes, f)

    return all_notes


if os.path.exists("data/notes.json"):
    with open('data/notes.json', 'r') as f:
        notes = json.load(f)
else:
    notes = get_notes()

model = NeuralNetwork(notes)
model.train()

