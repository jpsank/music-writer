import pretty_midi
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from model import NeuralNetwork
from util import extract_notes, load_json, save_json
from config import *


def extract_all(files):
    """ Extract midi input files to notes representation """

    results = []

    with ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(extract_notes, file): i
            for i, file in enumerate(files)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            res = future.result()
            results.extend(res)
            print(f"{idx}/{len(files)}, {len(results)} total")

    return results


if __name__ == '__main__':
    if os.path.exists(SAVE_NOTES):
        print("Loading saved notes...")
        notes = load_json(SAVE_NOTES)
    else:
        print("Extracting notes from midi files...")
        notes = extract_all(INPUT_FILES)

        print("Saving notes...")
        save_json(notes, SAVE_NOTES)

    print("Training...")
    print()

    model = NeuralNetwork(notes)
    model.train()
