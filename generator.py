import numpy as np
from tensorflow import keras
import random

from utils import TrackFrameSampler

np.seterr(divide='ignore', invalid='ignore')

class AudioDataGenerator(keras.utils.Sequence):
    """ Generates data for Keras. Enables on-the-fly augmentation and such.
    
    If ``batch_by_track`` is true, batches audio by track. This is useful
        when training needs to happen with samples in sequential order. For
        example, LSTMs need this, because they need to condition on the state
        outputted from the previous sample in time.
        
    ``tracks`` is a List<dataloader.Track> representing the tracks to sample from
    
    ``include_num_notes_played`` is true if the model takes two inputs: the raw waveform
        in addition to the number of notes played within that waveform. This is useful for
        models that predict notes from the waveform and the number of expected notes, as
        opposed to models that only look at the raw waveform.
    
    ``nsynth_chord_ratio`` determines the number of NSynth chords to generate
        per 'real' datapoint. 2.0 would do 2 chords per input, 1.0 would do 1...
    """
    def __init__(self, 
            tracks, frame_length, max_polyphony,
            randomize_train_frame_offsets=False,
            batch_size=32, 
            batch_by_track=False, 
            nsynth_chord_generator=None, 
            nsynth_chord_ratio=None, 
            normalize_audio=True,
            augmenter=None,
            sample_rate=16000,
            min_midi=25, max_midi=84,
            include_num_notes_played=False,
            label_format='categorical',
        ):
        self.label_format = label_format
        if label_format == 'categorical_and_input':
            label_format = 'categorical'
        elif label_format is None:
            label_format = 'midi_array'
        self.track_sampler = TrackFrameSampler(tracks, frame_length, batch_size, 
            label_format, min_midi, max_midi, max_polyphony,
            randomize_train_frame_offsets=randomize_train_frame_offsets, 
            batch_by_track=batch_by_track)
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.batch_by_track = batch_by_track
        self.normalize_audio = normalize_audio
        self.include_num_notes_played = include_num_notes_played
        self.on_epoch_end()
        if nsynth_chord_generator:
            if self.batch_by_track:
                assert batch_by_track == nsynth_chord_generator.batch_by_track, "either both generators must batch by track, or neither"
            else:
                assert batch_size == nsynth_chord_generator.batch_size, "data must be generated with consistent batch size"
        self.nsynth_chord_generator = nsynth_chord_generator
        self.nsynth_chord_ratio = nsynth_chord_ratio
        
        self.augmenter = augmenter
        if self.augmenter is not None:
            print('AudioDataGenerator using data augmentation')

    def __len__(self):
        """Denotes the number of batches per epoch."""
        num_batches = len(self.track_sampler)
        
        if self.nsynth_chord_generator:
            num_nsynth_chords = round(num_batches * self.nsynth_chord_ratio) \
                if self.nsynth_chord_ratio else len(self.nsynth_chord_generator)
            num_batches += num_nsynth_chords
        else:
            num_nsynth_chords = 0
        return num_batches

    def __getitem__(self, i, get_info=False):
        if i < len(self.track_sampler):
            x, y, info = self.track_sampler.__getitem__(i, get_info=get_info)
        else:
            x, y = next(self.nsynth_chord_generator)
            info = None # TODO implement for nsynth_chord_generator
        
        if self.augmenter:
            x = np.apply_along_axis(lambda w: self.augmenter(w, self.sample_rate), 1, x)
        
        if self.batch_by_track:
            num_samples_in_track, frame_length = x.shape
            if num_samples_in_track > self.batch_size:
                # When x is too wide for batch size, just reshape it to add a
                # batch dimension. This avoids having long dependencies for
                # really long tracks. Later, we can work on combining these 
                # segments in larger batches with other tracks (TODO).
                #
                # And yes, this loses a bit of temporal information (from the
                # 32nd to 33rd segment, for example, with batch size 32) but,
                # hey, what can you do...
                #
                # Generate zeros to make `x` the right shape
                num_zero_samples = self.batch_size - (num_samples_in_track % self.batch_size)
                x_zeros = np.zeros((num_zero_samples, frame_length))
                # Concatenate zeros and reshape x
                x = np.concatenate((x, x_zeros), axis=0)
                x = x.reshape((-1, self.batch_size, frame_length))
                # Reshape y to match
                num_samples, y_dim = y.shape
                y_zeros = np.zeros((num_zero_samples, y_dim))
                y = np.concatenate((y, y_zeros), axis=0)
                y = y.reshape((-1, self.batch_size, y_dim))
            else:
                # If x is small, just fit it in a single batch
                x, y = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
        
        # Normalize, potentially
        if self.normalize_audio:
            x = x - x.mean(axis=1)[:, np.newaxis]
            x = x / x.std(axis=1)[:, np.newaxis]
        
        # If number of notes played should be in input, get that
        if self.include_num_notes_played:
            num_notes_played = y.sum(axis=-1)
            x = { 
                'num_notes_played_input': num_notes_played, 
                'waveform_input': x
            }
        
        if self.label_format is None:
            if get_info:
                return x, x, info
            else:
                return (x, x)
        elif self.label_format == 'categorical_and_input':
            output_dict = { 'output_1': x,  'output_2': y }
            if get_info:
                return x, output_dict, info
            else:
                return x, output_dict
        else:
            if get_info:
                return x, y, info
            else:
                return x, y

    def on_epoch_end(self):
        """ Re-shuffle examples after each epoch. """
        self.track_sampler.on_epoch_end()
    
    def _callable(self, num_epochs):
        """ TensorFlow tf.data.Dataset.from_generator is overly picky about
        parameters. See https://stackoverflow.com/questions/49280016/how-to-make-a-generator-callable
        """
        def gen():
            for _ in range(num_epochs):
                for tuple_of_things in self:
                    yield tuple_of_things
        return gen


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
        include_num_notes_played=True,)
    idx = random.choice(range(len(gen)))
    x, y = gen[idx]
    # breakpoint()
    print('x.shape:', x.shape, 'y.shape:', y.shape)