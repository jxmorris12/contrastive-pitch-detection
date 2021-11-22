import collections
from typing import List, Tuple
import random

import numpy as np
import torch

from dataloader import MusicDataLoader
from dataloader.utils import Track
from utils import TrackFrameSampler

np.seterr(divide='ignore', invalid='ignore')

class AudioDataGenerator(torch.utils.data.Dataset):
    """ Generates data. Enables on-the-fly augmentation and such.

    This is a torch map-style dataset. See this link for more info:
        pytorch.org/docs/stable/data.html#map-style-datasets
    
    If ``batch_by_track`` is true, batches audio by track. This is useful
        when training needs to happen with samples in sequential order. For
        example, LSTMs need this, because they need to condition on the state
        outputted from the previous sample in time.
        
    ``tracks`` is a List<dataloader.Track> representing the tracks to sample from
    """
    def __init__(self, 
            tracks: List[Track], frame_length, max_polyphony,
            randomize_train_frame_offsets=False,
            batch_size=32, 
            batch_by_track=False,
            normalize_audio=True,
            augmenter=None,
            sample_rate=16000,
            min_midi=25, max_midi=84,
            label_format='categorical',
            num_fake_nsynth_chords=0,
        ):
        self.label_format = label_format
        if label_format is None:
            label_format = 'midi_array'
        if num_fake_nsynth_chords:
            print(f'Replacing {len(tracks)} tracks with {num_fake_nsynth_chords} fake NSynth chords')
            tracks = NSynthChordFakeTrackList(sample_rate, frame_length, num_fake_nsynth_chords)
        self.track_sampler = TrackFrameSampler(tracks, frame_length, batch_size, 
            label_format, min_midi, max_midi, max_polyphony,
            randomize_train_frame_offsets=randomize_train_frame_offsets, 
            batch_by_track=batch_by_track)
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.batch_by_track = batch_by_track
        self.normalize_audio = normalize_audio
        
        self.augmenter = augmenter
        if self.augmenter is not None:
            print('AudioDataGenerator using data augmentation')

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return len(self.track_sampler)

    def _get_track(self, i: int, get_info: bool) -> Tuple:
        """Gets track `i`. Can be overridden by subclasses."""
        return self.track_sampler.__getitem__(i, get_info=get_info)

    def __getitem__(self, i: int, get_info=False):
        x, y, info = self._get_track(i, get_info)
        if self.augmenter:
            x = np.apply_along_axis(lambda w: self.augmenter(w, self.sample_rate), 1, x)
        
        # Normalize, potentially
        if self.normalize_audio:
            x = x - x.mean(axis=1)[:, np.newaxis]
            x = x / x.std(axis=1)[:, np.newaxis]
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        if self.label_format is None:
            if get_info:
                return x, info
            else:
                return x
        else:
            if get_info:
                return x, y, info
            else:
                return x, y

    def on_epoch_end(self):
        """ Re-shuffle examples after each epoch. """
        self.track_sampler.on_epoch_end()

class NSynthChordFakeTrackList:
    """Generates data by adding NSynth notes together to make chords."""
    
    def __init__(self, sample_rate, frame_length, num_chord_batches: int):
        self.num_chord_batches = num_chord_batches
        self.notes_by_midi = collections.defaultdict(list)
        tracks = MusicDataLoader(sample_rate, frame_length, 
            # datasets=['nsynth_keyboard_train'],
            datasets=['nsynth_keyboard_valid'],
            batch_by_track=False, val_split=0.0
        ).load()
        for idx in range(len(tracks)):
            track = tracks[idx]
            instrument_str, midi, velocity = track.name.split('-')
            midi, velocity = int(midi), int(velocity)
            self.notes_by_midi[midi].append(track.waveform)
        self.notes_by_midi = { m: np.vstack(w) for m, w in self.notes_by_midi.items() }

    def __len__(self) -> int:
        """Denotes the number of batches per epoch."""
        return self.num_chord_batches
        
    @property
    def _midi_span(self):
        """ The number of MIDI notes to sample from. """
        return (self.max_midi - self.min_midi + 1)
    
    def get_chord_waveform(self, chord_midi_vals):
        # get waveforms and add them together
        # TODO add feature to get most popular chord from a file
        waveforms = [random.choice(self.notes_by_midi[m]) for m in chord_midi_vals if m > 0]
        waveforms = np.array(waveforms)
        i1 = np.random.choice(self._waveform_length)
        i2 = i1 + self.frame_length
        waveforms = waveforms[:, i1 : i2]
        chord_waveform = waveforms.sum(axis=0)
        # pad with zeros to meet frame_length
        if len(chord_waveform) < self.frame_length:
            num_zeros = self.frame_length - len(chord_waveform)
            left_padding = np.random.choice(num_zeros)
            right_padding = num_zeros - left_padding
            chord_waveform = np.pad(chord_waveform, (left_padding, right_padding),
                'constant', constant_values=(0,0))
        return chord_waveform

    def _get_track(self):
        """ Returns the batch at index <i>. """
        # Choose a note to get chords for at each index
        batch_midi_idxs = np.random.choice(self._midi_span, size=self.batch_size, p=self._midi_probs)
        midis = self._midis[batch_midi_idxs]
        
        # Get chord MIDIs and make categorical vector
        # TODO is apply_along_axis the right way to map a function over a 1D np array..?
        chord_midis = get_chord_from_midi_v(midis)
        if self.label_format == 'categorical':
            y = np.apply_along_axis(lambda _m: utils.midi_vals_to_categorical(_m, self.min_midi, self.max_midi), 1, chord_midis)
        elif self.label_format == 'midi_array':
            y = np.apply_along_axis(lambda _m: utils.midi_vals_to_midi_array(_m, self.min_midi, self.max_midi), 1, chord_midis)
        else:
            raise ValueError(f'unsupported label format {self.label_format}')
        
        # Create chords from MIDIs
        x = np.apply_along_axis(self.get_chord_waveform, 1, chord_midis)
        
        # If batching by track, 'unroll'
        if self.batch_by_track:
            x = np.squeeze(x)
            y = np.repeat(y[:, np.newaxis], len(x), 1)
            y = np.squeeze(y)
        return x, y

    def on_epoch_end(self):
        # Nothing to do here
        pass

if __name__ == '__main__':
    import random
    from dataloader import MusicDataLoader
    
    sr = 8000   # sample rate
    n = 512     # sample length
    
    data_loader = MusicDataLoader(sr, n, 
        min_midi=25, max_midi=85,
        label_format="categorical",
        datasets=['idmt_tiny'],
        normalize_audio=True,
        batch_by_track=True,
    )
    x, y = data_loader.load()
    gen = AudioDataGenerator(x, y, batch_by_track=True, sample_rate=sr,
        augmenter=None,
    )
    idx = random.choice(range(len(gen)))
    x, y = gen[idx]
    # breakpoint()
    print('x.shape:', x.shape, 'y.shape:', y.shape)