from typing import List, Tuple

import collections
import pickle
import random

import numpy as np
import torch

from dataloader import MusicDataLoader
from dataloader.utils import AnnotatedAudioChunk, Track, midi_to_hz
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
            tracks = NSynthChordFakeTrackList(
                num_fake_nsynth_chords, batch_size, sample_rate, frame_length,
                min_midi=min_midi, max_midi=max_midi
            )
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
    
    def __init__(self, num_chords_per_epoch: int, batch_size: int,
            sample_rate: int, frame_length: int,
            min_midi: int, max_midi: int,
            random_chords=True # Either random chords or top-K chords (calculated from MAESTRO dataset)
        ):
        self.num_chords_per_epoch = num_chords_per_epoch
        self.batch_size = batch_size
        self.min_midi = min_midi
        self.max_midi = max_midi
        tracks = MusicDataLoader(sample_rate, frame_length, 
            datasets=['nsynth_train'],
            # datasets=['nsynth_keyboard_train'],
            # datasets=['nsynth_keyboard_valid'],
            batch_by_track=False, val_split=0.0
        ).load()
        self.notes_by_midi = collections.defaultdict(list)
        for idx in range(len(tracks)):
            track = tracks[idx]
            instrument_str, midi, velocity = track.name.split('-')
            midi, velocity = int(midi), int(velocity)
            self.notes_by_midi[midi].append(track)
        self._midis = np.array([m for m in self.notes_by_midi.keys() if self.min_midi <= m <= self.max_midi])
        self._midis = np.sort(self._midis)
        self._midi_probs = np.array([len(self.notes_by_midi[m]) for m in self._midis])
        self._midi_probs = self._midi_probs.astype(float)
        self._midi_probs /= self._midi_probs.sum() # Normalize probabilities
        
        self.random_chords = random_chords
        if not self.random_chords:
            chords_by_num_notes = pickle.load(open('assets/maestrov3_chords_raw.p', 'rb'))
            self.top_chords = list(map(eval, chords_by_num_notes.keys()))
            print(f'NSynthChordFakeTrackList using top {len(self.top_chords)} chords')

    def __len__(self) -> int:
        """Denotes the number of batches per epoch."""
        return self.num_chords_per_epoch
        
    @property
    def _midi_span(self):
        """ The number of MIDI notes to sample from. """
        return (self.max_midi - self.min_midi + 1)

    def __getitem__(self, i) -> Track:
        """ Returns the batch at index <i>. """
        # Choose a note to get chords for at each index
        # TODO add feature to get most popular chord from a file
        if self.random_chords:
            # TODO(jxm): add argparse/settings that control flags, like the geometric dist on notes here
            # TODO(jxm): choose max polyphony based on args.max_polyphony argument
            #
            # geometric p=0.5
            # num_notes = np.random.choice([1,2,3,4,5,6], p=[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125])
            #
            # geometric p=0.8
            # num_notes = np.random.choice([1,2,3,4,5,6], p=[0.8000512032770098, 0.16001024065540193, 0.03200204813108038, 0.006400409626216074, 0.0012800819252432147, 0.00025601638504864284])
            #
            # uniform
            num_notes = np.random.choice([1,2,3,4,5,6])
            batch_midi_idxs = np.random.choice(self._midi_span, size=num_notes, p=self._midi_probs)
            midis = [m for m in self._midis[batch_midi_idxs] if m > 0]
        else:
            midis = random.choice(self.top_chords)
        tracks = [random.choice(self.notes_by_midi[m]) for m in midis]
        waveform = np.vstack([t.waveform for t in tracks]).sum(0) # sum along batch dimension

        chord_audio_chunk = AnnotatedAudioChunk(
            0, len(waveform), 
            tracks[0].sample_rate, 
           [midi_to_hz(m) for m in midis], [0], [1]
        )
        chord_track_name = '--'.join((t.name for t in tracks))
        dataset_names = sorted(set((t.dataset_name for t in tracks)))
        chord_dataset_name = '--'.join(dataset_names)
        return Track(
            chord_dataset_name, chord_track_name,
            [chord_audio_chunk], waveform,
            tracks[0].sample_rate, 
            name=chord_track_name
        )

    def on_epoch_end(self):
        # Nothing to do here
        pass
