import numpy as np
import pretty_midi

midi_file_name = "test.midi"
fs = 32

midi_pretty_format = pretty_midi.PrettyMIDI(midi_file_name)
piano_midi = midi_pretty_format.instruments[0] # Get the piano channels
piano_roll = piano_midi.get_piano_roll(fs=fs)

print((piano_roll.shape))
print(piano_roll[56][1:1000])