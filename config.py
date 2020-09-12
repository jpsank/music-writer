
import glob


# ------------ Train ------------

# INPUT_FILES = glob.glob("data/midi/*.mid")
# INPUT_FILES = glob.glob("data/midi/**/*.mid")
# INPUT_FILES = ["data/midi/Pokemon XY - Battle Wild Pokemon.mid"]
# INPUT_FILES = ["data/midi/Fugue1.mid"]
INPUT_FILES = ["data/midi/beethoven/firstmvm.mid", "data/midi/beethoven/secondmv.mid",
               "data/midi/beethoven/thirdmvm.mid"]
SAVE_NOTES = "data/notes.json"


# ------------ Generate ------------

WEIGHTS = 'weights-improvement-05-4.1350-bigger.hdf5'
TEMPO = 120  # output tempo
START = 0
NUM_PREDICTION = 200
OUTPUT = "test_output.mid"
PRIMER_MIDI = "data/midi/OmensOfLove.mid"
