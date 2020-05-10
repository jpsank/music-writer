from model import NeuralNetwork
from util import decode_representation, load_notes
from config import *


print("Loading saved notes...")
notes = load_notes()

model = NeuralNetwork(notes)
print("Loading weights...")
model.load_weights(WEIGHTS)

print("Generating prediction...")

prediction_output = model.predict(NUM_PREDICTION, START)

print("Prediction output:", prediction_output)
print("Creating midi...")

midi = decode_representation(prediction_output, tempo=TEMPO)
midi.write(OUTPUT)

print("Done")
