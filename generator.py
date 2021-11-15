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
    
    ``nsynth_chord_ratio`` determines the number of NSynth chords to generate
        per 'real' datapoint. 2.0 would do 2 chords per input, 1.0 would do 1...
    """
    def __init__(self, 
            tracks, frame_length, max_polyphony,
            randomize_train_frame_offsets=False,
            batch_size=32, 
            batch_by_track=False,
            normalize_audio=True,
            augmenter=None,
            sample_rate=16000,
            min_midi=25, max_midi=84,
            label_format='categorical',
        ):
        self.label_format = label_format
        if label_format is None:
            label_format = 'midi_array'
        self.track_sampler = TrackFrameSampler(tracks, frame_length, batch_size, 
            label_format, min_midi, max_midi, max_polyphony,
            randomize_train_frame_offsets=randomize_train_frame_offsets, 
            batch_by_track=batch_by_track)
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.batch_by_track = batch_by_track
        self.normalize_audio = normalize_audio
        self.on_epoch_end()
        
        self.augmenter = augmenter
        if self.augmenter is not None:
            print('AudioDataGenerator using data augmentation')

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return len(self.track_sampler)

    def __getitem__(self, i, get_info=False):
        x, y, info = self.track_sampler.__getitem__(i, get_info=get_info)
        
        if self.augmenter:
            x = np.apply_along_axis(lambda w: self.augmenter(w, self.sample_rate), 1, x)
        
        # Normalize, potentially
        if self.normalize_audio:
            x = x - x.mean(axis=1)[:, np.newaxis]
            x = x / x.std(axis=1)[:, np.newaxis]
        
        if self.label_format is None:
            if get_info:
                return x, x, info
            else:
                return (x, x)
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
    )
    idx = random.choice(range(len(gen)))
    x, y = gen[idx]
    # breakpoint()
    print('x.shape:', x.shape, 'y.shape:', y.shape)