import json
import pretty_midi
from collections import defaultdict


def load_json(fp):
    with open(fp, 'r') as f:
        return json.load(f)


def save_json(obj, fp):
    with open(fp, 'w') as f:
        json.dump(obj, f)


def encode_representation(midi_data):
    add_notes = []

    midi_map = defaultdict(list)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = int(midi_data.time_to_tick(note.start) / 10)
            end = int(midi_data.time_to_tick(note.end) / 10)
            midi_map[start].append(["on", note, instrument])
            midi_map[end].append(["off", note, instrument])

    last_time = 0
    for time in sorted(midi_map.keys()):
        notes_in_time = [
            "rest:{}".format(time - last_time)
        ]
        for item in midi_map[time]:
            on, note, instrument = item
            notes_in_time.append("{}:{}:{}".format(on, note.pitch, instrument.program))

        add_notes += notes_in_time
        last_time = time

    if add_notes:
        return ["^"] + add_notes + ["$"]  # signify start and end of sample
    else:
        return []


def extract_notes(file):
    try:
        midi_data = pretty_midi.PrettyMIDI(file)
        return encode_representation(midi_data)
    except Exception as e:
        print("Error:", e)
        return []


def decode_representation(rep_sequence, tempo=120):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    timing_map = {}
    instruments = {}
    offset = 0
    for element in rep_sequence:
        command, *params = element.split(":")
        if command == "on":  # note on
            element = element[2:]
            timing_map[element] = offset
        elif command == "off":  # note off
            element = element[3:]
            pitch, instr_program = params
            pitch = int(pitch)
            instr_program = int(instr_program)

            if instr_program not in instruments:
                instr = pretty_midi.Instrument(program=instr_program)
                instruments[instr_program] = instr

            if element in timing_map:  # in case the model creates invalid midi
                new_note = pretty_midi.Note(velocity=100, pitch=pitch, start=timing_map[element], end=offset)
                instruments[instr_program].notes.append(new_note)
        elif command == "rest":  # element must be delay between note/chords
            ticks = int(params[0])
            offset += midi.tick_to_time(ticks * 10)
        elif element == "$":  # end of song, wait one second
            offset += 1

    for instr in instruments:
        print(instruments[instr].notes)
        midi.instruments.append(instruments[instr])

    return midi
