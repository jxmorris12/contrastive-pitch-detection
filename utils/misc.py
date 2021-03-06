from typing import Dict

import collections
import functools
import io
import itertools
import math
import random

import numpy as np
import torch
import tqdm

def midi_vals_to_categorical(midi_values, min_midi, max_midi):
    """ Converts a list of float midi values to a single categorical vector. """
    category = np.zeros(max_midi - min_midi + 1)
    
    midi_values = np.array(midi_values)
    midi_values = np.rint(midi_values).astype(int)
    midi_values = np.unique(midi_values)
    
    midi_values = midi_values[(min_midi <= midi_values) & (midi_values <= max_midi)]
    
    category[midi_values - min_midi] = 1
    
    return category

def midi_vals_to_midi_array(midi_values, min_midi, max_midi, pad_to_length=6):
    midis = [m for m in midi_values if (min_midi <= m <= max_midi)]
    midis = np.array(midis)
    midis = np.sort(midis) # take lowest MIDIs if > 6? 
    midis = midis[:pad_to_length]
    if len(midis) < pad_to_length:
        pad_size = pad_to_length - len(midis)
        midis = np.pad(midis, (0, pad_size), 'constant', constant_values=(0, 0))
    return midis

@functools.lru_cache
def hz_to_midi(f):
    if f == 0: 
        return 0
    else:
        return 69.0 + 12.0 * math.log2(f / 440.0)

hz_to_midi_v = np.vectorize(hz_to_midi)

@functools.lru_cache
def midi_to_hz(d):
    exp = (d - 69) / 12.0
    return (2 ** exp) * 440.0

FrameInfo = collections.namedtuple('FrameInfo', ['dataset_name', 'track_name', 'sample_rate', 'start_time', 'end_time'])

def combinations_2000(some_list, n):
    """Returns up to 2000 combinations from some_list choose n.
    """
    combos = itertools.combinations(some_list, n)
    i = 0
    for combo in combos:
        yield combo
        i += 1
        if i >= 2000:
            return

def note_and_neighbors(note_midi, min_midi, max_midi, num_random_notes=10):
    # TODO(jxm): Split this into multiple functions or something.
    # For now this is just returning all valid chords within two octaves
    # of the base note!
    # TODO(jxm): six_note_chords will be huge here if min_midi << max_midi, what to do then?
    neighbor_notes = list(range(note_midi-12, note_midi)) + list(range(note_midi+1, note_midi+12))
    neighbor_notes = [n for n in neighbor_notes if min_midi <= n <= max_midi]
    # Randomize the order of neighbor notes so that we get a random order of combinations.
    random.shuffle(neighbor_notes)
    one_note_chords =  [[note_midi]] + [[n] for n in neighbor_notes]
    two_note_chords = [[note_midi, n] for n in neighbor_notes]
    three_note_chords = [[note_midi] + list(chord) for chord in combinations_2000(neighbor_notes, 2)]
    four_note_chords = [[note_midi] + list(chord) for chord in combinations_2000(neighbor_notes, 3)]
    five_note_chords = [[note_midi] + list(chord) for chord in combinations_2000(neighbor_notes, 4)]
    six_note_chords = [[note_midi] + list(chord) for chord in combinations_2000(neighbor_notes, 5)]
    return {
        1: one_note_chords,
        2: two_note_chords,
        3: three_note_chords,
        4: four_note_chords,
        5: five_note_chords,
        6: six_note_chords,
    }

# def note_and_neighbors(note_midi, min_midi, max_midi, num_random_notes=10):
#     neighbor_notes = [
#         note_midi - 24, # two octaves down
#         note_midi - 12, # octave down
#         note_midi + 12, # octave up
#         note_midi + 24, # two octaves up
#         note_midi - 1, # note off-by-one
#         note_midi + 1, # note off-by-one
#         note_midi - 4, # perfect third
#         note_midi + 4, # perfect third
#         note_midi + 7, # perfect fifth
#         note_midi - 7, # perfect fifth
#     ]
#     # all_other_notes = set(range(min_midi, max_midi+1)) - set(neighbor_notes)
#     all_other_notes = set(range(note_midi-12, note_midi+12)) - set(neighbor_notes)
#     all_other_notes = [n for n in all_other_notes if min_midi <= n <= max_midi]
#     neighbor_notes += random.sample(all_other_notes, min(num_random_notes, len(all_other_notes)))
#     neighbor_notes = [n for n in neighbor_notes if min_midi <= n <= max_midi] # filter out notes that are too high or low
#     one_note_chords = [[note_midi]] + [[n] for n in neighbor_notes]
#     two_note_chords = [[note_midi, n] for n in neighbor_notes]
#     three_note_chords = [[note_midi] + list(chord) for chord in itertools.combinations(neighbor_notes, 2)]
#     four_note_chords = [[note_midi] + list(chord) for chord in itertools.combinations(neighbor_notes, 3)]
#     five_note_chords = [[note_midi] + list(chord) for chord in itertools.combinations(neighbor_notes, 4)]
#     six_note_chords = [[note_midi] + list(chord) for chord in itertools.combinations(neighbor_notes, 5)]
#     return { 
#         1: one_note_chords,
#         2: two_note_chords,
#         3: three_note_chords,
#         4: four_note_chords,
#         5: five_note_chords,
#         6: six_note_chords,
#     }

class TrackFrameSampler:
    """ Given a list of tracks of different lengths, how can you randomly
    sample N frames continuously while avoiding repetition?
    
    ``max_polyphony`` is the maximum number of notes played in a chord to be
        considered for training
        
    - ``label_format`` determines the format that the y vector pops out in.
        
        - 'categorical': the vector will be a list binned from 
            ``min_midi`` to ``max_midi`` where any MIDI values present in the
            audio are set to 1.
            
        - 'midi_array': output is an array of 6 [lowest?] MIDI values
        
        - 'cents': MIDI values are split into 360-length cents
            bins as done in this CREPE paper: 

    """
    def __init__(self, tracks, frame_length, batch_size, label_format, min_midi, 
        max_midi, max_polyphony, batch_by_track=False, skip_empty_samples=True,
        randomize_train_frame_offsets=False):
        assert batch_size > 0, 'batch size must be positive'
        self.tracks = tracks
        self.frame_length = frame_length
        self.batch_size = batch_size
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.max_polyphony = max_polyphony or float('inf')
        self.skip_empty_samples = skip_empty_samples
        self.batch_by_track = batch_by_track
        self.randomize_train_frame_offsets = randomize_train_frame_offsets
        self._init_track_frame_index_pairs()
        print(f'TrackFrameSampler loaded {len(self.track_frame_index_pairs)} frames')

        assert label_format in { "categorical", "cents", "midi_array" }
        self.label_format = label_format
        
        self.on_epoch_end()
    
    def __len__(self):
        if self.batch_by_track:
            return len(self.tracks)
        else:
            if len(self.track_frame_index_pairs) == 0:
                length = 0
            else:
                length = max(len(self.track_frame_index_pairs) // self.batch_size, 1)
            if length == 0:
                raise RuntimeError('TrackFrameSampler has length 0')
            return length
    
    def _init_track_frame_index_pairs(self):
        """ Frames are indexed by a tuple ``(track_idx, frame_idx)``. This builds
        a list of such tuples from ``self.tracks``.
        
        Also filters out empty samples (given ``self.skip_empty_samples`` is True)
            and filters out samples with too many notes being played as according
            to ``self.max_polyphony``.
        """
        # TODO(jxm): This only works with randomly generated data because the waveforms
        # all have the same shape, which is dumb. Need to do this better somehow.
        self.track_frame_index_pairs = []
        for track_idx in tqdm.trange(len(self.tracks), desc='TrackFrameSampler counting frames'):
            if hasattr(self.tracks, 'reset_for_next_batch'):
                # Ah, we shouldn't need to do this next line, it's an unfortunate
                # result of the cruft in this codebase.
                # TODO(jxm): refactor this away :-)
                self.tracks.reset_for_next_batch()
            track = self.tracks[track_idx]
            for frame_idx in range(track.num_frames(self.frame_length)):
                frequencies = track.get_frequencies_from_offset(frame_idx * self.frame_length, (frame_idx + 1) * self.frame_length)
                if not len(frequencies) and self.skip_empty_samples:
                    continue
                elif len(frequencies) > self.max_polyphony: 
                    continue
                else:
                    self.track_frame_index_pairs.append((track_idx, frame_idx))
        if len(self.track_frame_index_pairs) == 0:
            raise RuntimeError(f'Got 0 track_frame_index_pairs from {len(self.tracks)} tracks')
        
        if not self.batch_by_track:
            random.shuffle(self.track_frame_index_pairs)
        
    def frequencies_to_label(self, frequencies):
        """ Converts a list of frequencies to the proper format for training. """
        if not len(frequencies):
            midis = []
        else:
            midis = np.rint(hz_to_midi_v(frequencies))
        if self.label_format == 'categorical':
            categorical = midi_vals_to_categorical(midis, self.min_midi, self.max_midi)
            return categorical
        elif self.label_format == 'midi_array':
            midis = midi_vals_to_midi_array(midis, self.min_midi, self.max_midi)
            return midis
        elif self.label_format == 'cents':
            raise NotImplementedError('need to add code for calculating cents') 
        else:
            raise ValueError(f'unsupported label format {self.label_format}')
    
    def __getitem__(self, idx, get_info=False):
        """ Gets the next batch of audio. """
        if idx >= len(self):
            print('Raising stopiteration with idx', idx, 'and len(self) =', len(self))
            raise StopIteration
        batch_track_frame_index_pairs = self.track_frame_index_pairs[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        track_info = []
        for track_idx, frame_idx in batch_track_frame_index_pairs:
            track = self.tracks[track_idx]
            start_idx = frame_idx * self.frame_length
            if self.randomize_train_frame_offsets:
                # TODO(jxm): Make sure this is OK.
                # Note: By adding this random offset, we don't guarantee that
                # the waveform length is greater than 'min_presence', nor that
                # it contains any audible notes at all.
                rand_step = self.frame_length // 2
                start_idx += random.randint(-rand_step, rand_step)
                start_idx = max(start_idx, 0)
                start_idx = min(start_idx, len(track) - self.frame_length)
            end_idx = start_idx + self.frame_length
            waveform = track.waveform[start_idx : end_idx]
            frequencies = track.get_frequencies_from_offset(start_idx, end_idx)
            batch_x.append(waveform)
            batch_y.append(self.frequencies_to_label(frequencies))
            if get_info:
                sample_rate = track.sample_rate
                start_time = start_idx / sample_rate
                end_time = end_idx / sample_rate
                track_info.append(
                    FrameInfo(
                        dataset_name=track.dataset_name, track_name=track.track_name,
                        sample_rate=sample_rate, start_time=start_time, end_time=end_time
                    ))

        if get_info:
            return np.vstack(batch_x), np.vstack(batch_y), track_info
        else:
            return np.vstack(batch_x), np.vstack(batch_y), []
        
    def on_epoch_end(self):
        if self.batch_by_track:
            random.shuffle(self.tracks)
            self._init_track_frame_index_pairs()
        else:
            random.shuffle(self.track_frame_index_pairs)

def get_prediction_type(pred_midis, true_midis):
    """ Returns a string explaining what the model did wrong here. 

        options:
            [
                'correct_silence', 'silence_instead_of_notes', 'notes_instead_of_silence',
                'correct_single_note', 'correct_chord', 'incorrect_single_note',
                'incorrect_chord', 'overpredicted_correct_single_note',
                'overpredicted_correct_chord', 'overpredicted_incorrect_single_note',
                'overpredicted_incorrect_chord', 'underpredicted_correct_chord',
                'underpredicted_incorrect_chord', 'other'
            ]
        
        TODO add detail related to harmonics
    """
    pred_midi_set = set(pred_midis)
    true_midi_set = set(true_midis)
    if not len(pred_midis):
        if not len(true_midis):
            # correctly predicted no notes when there weren't any
            return 'correct_silence'
        else:
            # incorrectly predicted silence when there was a note
            return 'silence_instead_of_notes'
    elif not len(true_midis):
        # predicted notes when there should've been silence
        return 'notes_instead_of_silence'
    elif len(true_midis) == len(pred_midis):
        if np.array_equal(true_midis, pred_midis):
            if len(true_midis) == 1:
                # predicted single note correctly
                return 'correct_single_note'
            else:
                return 'correct_chord'
        else:
            if len(true_midis) == 1:
                return 'incorrect_single_note'
            else:
                return 'incorrect_chord'
    elif len(true_midis) < len(pred_midis):
        if true_midi_set.issubset(pred_midi_set):
            if len(true_midis) == 1:
                return 'overpredicted_correct_single_note'
            else:
                return 'overpredicted_correct_chord'
        else:
            if len(true_midis) == 1:
                return 'overpredicted_incorrect_single_note'
            else:
                return 'overpredicted_incorrect_chord'
    elif len(true_midis) > len(pred_midis):
        if pred_midi_set.issubset(true_midi_set):
            return 'underpredicted_correct_chord'
        else:
            return 'underpredicted_incorrect_chord'
    else:
        return 'other'


class TensorRunningAverages:
    _store_sum: Dict[str, torch.Tensor]
    _store_total: Dict[str, torch.Tensor]

    def __init__(self):
        self._store_sum = {}
        self._store_total = {}

    def update(self, key: str, val: torch.Tensor) -> None:
        if key not in self._store_sum:
            self.clear(key)
        self._store_sum[key] += val.detach().cpu()
        self._store_total[key] += 1

    def get(self, key: str) -> float:
        total = max(self._store_total.get(key).item(), 1.0)
        return (self._store_sum[key] / float(total)).item() or 0.0
    
    def clear(self, key: str) -> None:
        self._store_sum[key] = torch.tensor(0.0, dtype=torch.float32)
        self._store_total[key] = torch.tensor(0, dtype=torch.int32)
    
    def clear_all(self) -> None:
        for key in self._store_sum:
            self.clear(key)

