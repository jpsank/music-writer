from model import NeuralNetwork
from util import decode_representation, load_json, extract_notes
from config import *


print("Loading saved notes...")
notes = load_json(SAVE_NOTES)

model = NeuralNetwork(notes)
print("Loading weights...")
model.load_weights(WEIGHTS)

if PRIMER_MIDI:
    print("Loading primer...")
    primer_notes = extract_notes(PRIMER_MIDI)
    model.set_primer(primer_notes)

print("Generating prediction...")

prediction_output = model.predict(NUM_PREDICTION)

print("Prediction output:", prediction_output)
print("Creating midi...")

midi = decode_representation(prediction_output, tempo=TEMPO)
midi.write(OUTPUT)

print("Done")
