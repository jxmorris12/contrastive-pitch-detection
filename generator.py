from typing import List, Union, Tuple

import collections
import pickle
import random

import numpy as np
import torch

from dataloader import MusicDataLoader
from dataloader.utils import AnnotatedAudioChunk, Track, midi_to_hz
from utils import TrackFrameSampler, note_and_neighbors

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
        self.num_fake_nsynth_chords = num_fake_nsynth_chords
        if num_fake_nsynth_chords:
            print(f'Replacing {len(tracks)} tracks with {num_fake_nsynth_chords} fake NSynth chords')
            tracks = NSynthChordFakeTrackList(
                num_fake_nsynth_chords, batch_size, sample_rate, frame_length, max_polyphony,
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
        
        self.min_midi = min_midi
        self.max_midi = max_midi

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return len(self.track_sampler)

    def _get_track(self, i: int, get_info: bool) -> Tuple:
        """Gets audio for batch `i`."""
        return self.track_sampler.__getitem__(i, get_info=get_info)

    def __getitem__(self, i: int, get_info=False):
        """Gets batch `i` from `self.track_sampler`."""
        if self.num_fake_nsynth_chords:
            # TODO(jxm): refactor so this is all less hacky!
            self.track_sampler.tracks.reset_for_next_batch()
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
    """Generates data by adding NSynth notes together to make chords.
    
    Then pretends to be a List[Track].
    """
    
    def __init__(self, num_chords_per_epoch: int, batch_size: int,
            sample_rate: int, frame_length: int, max_polyphony: float,
            min_midi: int, max_midi: int,
            random_chords=True # Either random chords or top-K chords (calculated from MAESTRO dataset)
        ):
        self.num_chords_per_epoch = num_chords_per_epoch
        self.batch_size = batch_size
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.max_polyphony = max_polyphony
        tracks = MusicDataLoader(sample_rate, frame_length, 
            # datasets=['nsynth_train'],
            datasets=['nsynth_keyboard_train'],
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
        # Stores the chords that can be sampled from if not random. TODO(jxm): refactor to do this better.
        self.chords_to_sample_from = {} # maps note_num -> midis, like { 1: [[14, 17, 20], ...], ...}

    def __len__(self) -> int:
        """Denotes the number of batches per epoch."""
        return self.num_chords_per_epoch
        
    @property
    def _midi_span(self):
        """ The number of MIDI notes to sample from. """
        return (self.max_midi - self.min_midi + 1)
    
    def reset_for_next_batch(self):
        """Resets stuff in between batches."""
        # If we're using the fake NSynth training data, we should update the NSynthChordFakeTrackList
        # (which is self.track_sampler.tracks) so that it can generate a batch of fake chords with 
        # neighbors.
        # TODO(jxm): I really need to refactor, this part is getting bad.
        random_note = np.random.choice(range(self.min_midi, self.max_midi+1))
        self.chords_to_sample_from = note_and_neighbors(
            random_note, self.min_midi, self.max_midi)

    def _get_track_from_chord_midis(self, midis: Union[List[int], np.ndarray]) -> Track:
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

    @property
    def _valid_note_nums(self) -> Tuple[np.ndarray, np.ndarray]:
        """Valid numbers of notes for generation and their probabilities."""
        notes = np.array([1,2,3,4,5,6])
        # geometric p=0.5
        note_probs = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125])
        # geometric p=0.8
        # note_probs = np.array([0.8000512032770098, 0.16001024065540193, 0.03200204813108038, 0.006400409626216074, 0.0012800819252432147, 0.00025601638504864284])
        # uniform
        # note_probs = np.array([0.166666666666666, 0.166666666666666, 0.166666666666666, 0.166666666666666, 0.166666666666666, 0.166666666666666])

        # Remove potential chords with too many notes according to the `max_polyphony` argument
        #
        # Also remove numbers of notes that don't have any corresponding chords in
        # self.chords_to_sample_from. This could happen if we're sampling without
        # replacement and we sampled all the chords away.
        notes_valid = np.logical_and(
            (notes <= self.max_polyphony), 
            np.array([n in self.chords_to_sample_from for n in notes])
        )
        notes = notes[notes_valid]
        note_probs = note_probs[notes_valid]
        note_probs = note_probs / note_probs.sum() # Normalize to form a probability distribution

        return notes, note_probs

    def _sample_random_num_notes(self) -> int:
        """Samples a random number of notes to generate a chord.
        
        Will adjust number of available notes to respect the `max_polyphony` argument.
        Depends on 
        """
        if not self.chords_to_sample_from:
            # This can only happen if we don't have enough neighbors returned from the neighbor_fn
            # and the batch size is too high.
            raise RuntimeError('Ran out of chords to sample from.')
        notes, note_probs = self._valid_note_nums
        if not len(notes):
            # Ran out of notes to sample from, start recycling
            self.reset_for_next_batch()
            # Recursively refill - watch out for infinite loops!!
            return self._sample_random_num_notes()
        return np.random.choice(notes, p=note_probs)

    def __getitem__(self, i) -> Track:
        """ Returns the chord at index <i>. These chords are typically batched by the TrackFrameSampler."""
        # Choose a note to get chords for at each index
        # TODO add feature to get most popular chord from a file
        if self.chords_to_sample_from:
            # print(f'* * * NSynthChordFakeTrackList using {len(self.chords_to_sample_from)} chords')
            # num_notes = np.random.choice([1,2,3,4,5,6])
            num_notes = self._sample_random_num_notes()
            # Sample a random note.
            midis = random.choice(self.chords_to_sample_from[num_notes])
            # And sample without replacement.
            self.chords_to_sample_from[num_notes].remove(midis)
            if not len(self.chords_to_sample_from[num_notes]):
                del self.chords_to_sample_from[num_notes]
        elif self.random_chords:
            # print(f'* * * NSynthChordFakeTrackList generating random chords')
            # TODO(jxm): add argparse/settings that control flags, like the geometric dist on notes here
            # TODO(jxm): choose max polyphony based on args.max_polyphony argument
            #
            num_notes = self._sample_random_num_notes()
            #
            batch_midi_idxs = np.random.choice(self._midi_span, size=num_notes, p=self._midi_probs)
            midis = [m for m in self._midis[batch_midi_idxs] if m > 0]
        else:
            midis = random.choice(self.top_chords)
        return self._get_track_from_chord_midis(midis)

    def on_epoch_end(self):
        # Nothing to do here
        pass
