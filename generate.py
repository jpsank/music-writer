import numpy as np
import json, os

from model import decode_representation, NeuralNetwork


with open('data/notes.json', 'r') as f:
    notes = json.load(f)


model = NeuralNetwork(notes)
model.load_weights('weights-improvement-14-1.5380-bigger.hdf5')

print("Generating notes...")

prediction_output = model.predict(500)

print("Prediction output:", prediction_output)
print("Creating midi...")

midi = decode_representation(prediction_output, tempo=600)
midi.write('test_output.mid')

print("Done")
