import collections
import io
import functools
import math
import numpy as np
import random
import tensorflow as tf

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
            length = len(self.track_frame_index_pairs) // self.batch_size
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
        self.track_frame_index_pairs = []
        for track_idx, track in enumerate(self.tracks):
            for frame_idx in range(track.num_frames(self.frame_length)):
                frequencies = track.get_frequencies_from_offset(frame_idx * self.frame_length, (frame_idx + 1) * self.frame_length)
                if not len(frequencies) and self.skip_empty_samples:
                    continue
                elif len(frequencies) > self.max_polyphony: 
                    continue
                else:
                    self.track_frame_index_pairs.append((track_idx, frame_idx))
        
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
        batch_track_frame_index_pairs = self.track_frame_index_pairs[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        track_info = []
        for track_idx, frame_idx in batch_track_frame_index_pairs:
            track = self.tracks[track_idx]
            start_idx = frame_idx * self.frame_length
            if self.randomize_train_frame_offsets:
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

def pyplot_to_numpy(plt):
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    plt.cla()
    plt.close()
    return image

def get_prediction_type(pred_midis, true_midis):
    """ Returns a string explaining what the model did wrong here. 
        
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