import collections
from typing import List

import librosa
import os
import pathlib
import pickle
import tqdm

from .utils import midi_to_hz, AnnotatedAudioChunk, Track

# TODO(jxm): Add trim_silence as a global preprocessing option.
trim_silence = False
# TODO(jxm): look up instrument numbers to enable other options
INSTRUMENT_FAMILY_NUMS = { 'keyboard': 4 }
def _load_nsynth(split, instrument) -> List[Track]:
    import tensorflow_datasets as tfds

    assert split in {'train', 'test', 'valid'}
    SAMPLE_RATE = 16_000 # this is constant for NSynth

    print(f'Loading NSynth split {split} and instrument {instrument}')
   
    # Bad samples (that I know about so far...)
    bad_nsynth_insts = {
        'guitar_acoustic_023',
        'guitar_acoustic_025',
        'guitar_acoustic_031',
        'guitar_acoustic_036',
    }
    
    nsynth = tfds.audio.Nsynth()
    ds = nsynth.as_dataset()[split]
    if instrument:
        ds = ds.filter(
            lambda r: r['instrument']['family'] == INSTRUMENT_FAMILY_NUMS[instrument]
        )
    tracks = []
    total_length = 0
    file_prefixes = collections.Counter()
    total_count = 0
    for data in tqdm.tqdm(iter(ds), desc=f'Loading NSynth split {split}'):
        sample_name = str(data['id'])
        if sample_name in bad_nsynth_insts:
            print(f'Skipping bad sample {sample_name}')
            continue
        total_count += 1
        raw_waveform = data['audio'].numpy()
        # Trim leading and trailing silence.
        if trim_silence:
            raw_waveform, trimmed_region = librosa.effects.trim(raw_waveform, top_db=20)
        # TODO Fake string and fret number?
        midi = data['pitch'].numpy()
        # 
        # 
        # data sample: 
        # {
        # 'audio': <tf.Tensor: shape=(64000,), dtype=float32, numpy=array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)>, 
        # 'id': <tf.Tensor: shape=(), dtype=string, numpy=b'keyboard_electronic_054-106-100'>,
        # 'instrument': {'family': <tf.Tensor: shape=(), dtype=int64, numpy=4>,
        # 'label': <tf.Tensor: shape=(), dtype=int64, numpy=480>,
        # 'source': <tf.Tensor: shape=(), dtype=int64, numpy=1>},
        # 'pitch': <tf.Tensor: shape=(), dtype=int64, numpy=106>,
        # 'qualities': {
            # 'bright': <tf.Tensor: shape=(), dtype=bool, numpy=True>,
            # 'dark': <tf.Tensor: shape=(), dtype=bool, numpy=False>,
            # 'distortion': <tf.Tensor: shape=(), dtype=bool, numpy=False>,
            # 'fast_decay': <tf.Tensor: shape=(), dtype=bool, numpy=False>, 
            # 'long_release': <tf.Tensor: shape=(), dtype=bool, numpy=False>,
            # 'multiphonic': <tf.Tensor: shape=(), dtype=bool, numpy=False>, 
            # 'nonlinear_env': <tf.Tensor: shape=(), dtype=bool, numpy=False>,
            # 'percussive': <tf.Tensor: shape=(), dtype=bool, numpy=False>,
            # 'reverb': <tf.Tensor: shape=(), dtype=bool, numpy=False>,
            # 'tempo-synced': <tf.Tensor: shape=(), dtype=bool, numpy=False>
        # }, 
        # 'velocity': <tf.Tensor: shape=(), dtype=int64, numpy=100>
        # }
        # 
        # 
        freq = midi_to_hz(midi)
        new_sample = AnnotatedAudioChunk(
            0, len(raw_waveform), 
            SAMPLE_RATE, 
            [freq], [0], [1]
        )
        total_length += new_sample.length_in_seconds
        track = Track('nsynth', sample_name, [new_sample], raw_waveform, SAMPLE_RATE, name=sample_name)
        tracks.append(track)
    print(f'NSynth ({split},{instrument}) total number of samples: {total_count}')
    print(f'NSynth ({split},{instrument}) loaded {total_length:.2f}s of audio ({len(tracks)} tracks)')
    return tracks

def load_nsynth(*args):
    folder = pathlib.Path(__file__).resolve().parent
    params_key = '_'.join(str(p) for p in args)
    cached_file = os.path.join(folder, '.nsynth_cache', params_key + '.p')
    # make cache if it doesn't exist
    (
        pathlib.Path(os.path.join(folder, '.nsynth_cache'))
        .mkdir(parents=True, exist_ok=True)
    )
    # Get data and save to cache
    if os.path.exists(cached_file):
        return pickle.load(open(cached_file, 'rb'))
    else:
        data = _load_nsynth(*args)
        pickle.dump(data, open(cached_file, 'wb'))
        print('Wrote NSynth data to', cached_file)
        return data