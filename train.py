import pretty_midi
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from model import NeuralNetwork
from util import encode_representation, load_notes, save_notes
from config import *


def extract_notes(file):
    try:
        midi_data = pretty_midi.PrettyMIDI(file)
        return encode_representation(midi_data)
    except Exception as e:
        print("Error:", e)
        return []


def extract_all():
    """ Extract midi input files to notes representation """

    results = []

    with ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(extract_notes, file): i
            for i, file in enumerate(INPUT_FILES)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            res = future.result()
            results.extend(res)
            print(f"{idx}/{len(INPUT_FILES)}, {len(results)} total")

    return results


if __name__ == '__main__':
    if os.path.exists(SAVE_NOTES):
        print("Loading saved notes...")
        notes = load_notes()
    else:
        print("Extracting notes from midi files...")
        notes = extract_all()

        print("Saving notes...")
        save_notes(notes)

    print("Training...")
    print()

    model = NeuralNetwork(notes)
    model.train()
