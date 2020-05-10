import json
import pretty_midi

from config import *


def load_notes():
    with open(SAVE_NOTES, 'r') as f:
        return json.load(f)


def save_notes(notes):
    with open(SAVE_NOTES, 'w') as f:
        json.dump(notes, f)


def encode_representation(midi_data):
    add_notes = []

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
            "rest{}".format(time - last_time)
        ]
        for item in midi_map[time]:
            on, note, instrument = item
            notes_in_time.append("{}{}-{}".format(on, note.pitch, instrument.program))

        add_notes += notes_in_time
        last_time = time

    if add_notes:
        return ["^"] + add_notes + ["$"]  # signify start and end of sample
    else:
        return []


def decode_representation(rep_sequence, tempo=120):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    timing_map = {}
    instruments = {}
    offset = 0
    for element in rep_sequence:
        if element.startswith("on"):  # note on pointer
            element = element[2:]
            timing_map[element] = offset
        elif element.startswith("off"):  # note off pointer
            element = element[3:]
            pitch, instr_program = element.split("-")
            pitch = int(pitch)
            instr_program = int(instr_program)

            if instr_program not in instruments:
                instr = pretty_midi.Instrument(program=instr_program)
                instruments[instr_program] = instr

            if element in timing_map:  # in case the model creates invalid midi
                new_note = pretty_midi.Note(velocity=100, pitch=pitch, start=timing_map[element], end=offset)
                instruments[instr_program].notes.append(new_note)
        else:
            if element == "$":  # end of song, wait one second
                offset += 1
            elif element.startswith("rest"):  # element must be delay between note/chords
                element = element[4:]
                offset += midi.tick_to_time(int(element) * 10)

    for instr in instruments:
        print(instruments[instr].notes)
        midi.instruments.append(instruments[instr])

    return midi
