import collections
import functools
import glob
import interlap
import json
import math
import nicecache
import numpy as np
import os
import pickle
import random
import torchaudio
import tqdm

from utils import midi_vals_to_categorical, midi_vals_to_midi_array, hz_to_midi, midi_to_hz

import librosa
import methodtools
from mirdata import guitarset
import scipy.signal as sps

class AnnotatedAudioChunk:
    """ An audio snippet. Consists of an audio sample, frequency annnotations,
        and fret/string number annotations. 
        
        - ``start_idx`` is the index this chunk begins in the initial waveform
        - ``end_idx`` is the index this chunk ends in the initial waveform
        - ``sample_rate`` is the sampling rate of the initial waveform
        - ``frequencies``, ``fret_numbers``, ``string_numbers`` are properties
            of the annotations of the original waveform
    """
    def __init__(self, start_idx, end_idx, sample_rate, frequencies, fret_numbers, string_numbers):
        assert start_idx <= end_idx, "start must come before end"
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.sample_rate = sample_rate
        assert sample_rate > 0, "sample rate must be positive"
        # frequencies
        assert len(frequencies) > 0, "must have at least one frequency annotation"
        for frequency in frequencies:
            # lowest frequency on a bass is 41
            # highest frequency on a 20-note guitar is 1047
            # maybe 24-note guitar gets up to 1245?
            # somehow this dataset gets much higher. at least up to MIDI 100 / 2637. Hz
            #
            # Ok, NSynth goes from MIDI 25, or 27.5 Hz,
            #   up to at least 7040 Hz...
            #
            # But NSynth BASS goes as low as MIDI 16, which is 20.6 Hz!!
            #
            assert 0 < frequency <= 9000, f"must be a valid guitar frequency (got {frequency})"
        self.frequencies = frequencies
        # fret number
        for fret_number in fret_numbers:
            assert 0 <= fret_number <= 24, "fret number must be in [0, 24]"
        self.fret_numbers = fret_numbers
        # string number
        assert len(string_numbers) > 0, "must have at least one string annotation"
        for string_number in string_numbers:
            assert 1 <= string_number <= 6, "string number must be in [1, 6]"
        self.string_numbers = string_numbers
    
    def __repr__(self):
        return f"""<AnnotatedAudioChunk length={self.length_in_seconds:.2f}s midis={self.midi_values}>"""
    
    __str__ = __repr__
    
    @property
    def length_in_seconds(self):
        return (self.end_idx - self.start_idx) / self.sample_rate
        
    @property
    def midi_values(self):
        return list(map(hz_to_midi, self.frequencies))
    
    @staticmethod
    def merge_samples(samples):
        sample_rate = samples[0].sample_rate
        # TODO throw error if some samples have mismatched sampling rates
        start_idx = min((s.start_idx for s in samples))
        end_idx = max((s.end_idx for s in samples))
        frequencies = [f for s in samples for f in s.frequencies]
        fret_numbers = [f for s in samples for f in s.fret_numbers]
        string_numbers = [f for s in samples for f in s.string_numbers]
        return AnnotatedAudioChunk(start_idx, end_idx, sample_rate, frequencies, fret_numbers, string_numbers)
        
    def split(self, waveform, new_sample_rate, sample_length, include_partial_chunk=True):
        """ Resamples and splits into audio chunks of length ``sample_length``.
            
            If ``include_partial_chunk`` is True, and the last chunk is shorter
                than ``sample_length``, pads with zeros and returns it.
        """
        if new_sample_rate != self.sample_rate:
            data = self.resample_waveform(waveform, new_sample_rate)
        
        if len(data) % sample_length == 0:
            num_chunks = int(len(data) / sample_length)
            chunks = np.split(data, num_chunks)
        else:
            leftover = len(data) % sample_length
            last_chunk = data[-leftover:]
            pad_size = sample_length - leftover
            last_chunk = np.pad(last_chunk, (0, pad_size), 'constant', constant_values=(0, 0))
            data = data[:-leftover]
            num_chunks = int(len(data) / sample_length)
            chunks = np.split(data, num_chunks) if num_chunks > 0 else []
            if include_partial_chunk:
                chunks += [last_chunk]
        
        return chunks
    
    def sample(self, waveform, new_sample_rate, sample_length):
        """" Returns a random audio sample. Resamples and gets a random portion of length ``length``. """
        data = self.resample_waveform(waveform, new_sample_rate)
        data_start = random.randint(0, len(data) - sample_length)
        data_end = data_start + sample_length
        return data[data_start : data_end]

class Track:
    """ A guitar track is a list of ``Chunk`` objects that come from the
        same recording.
        
        It detects which samples overlap in a given audio sample.
    """
    def __init__(self, dataset_name, track_name, samples, waveform, sample_rate, name=None):
        """ ``samples`` is a List<AnnotatedAudioChunk>
        
        See https://github.com/brentp/interlap
        """
        self.dataset_name = dataset_name
        self.track_name = track_name
        self.interval = interlap.InterLap()
        self.waveform = waveform
        self.original_sample_rate = sample_rate
        self.sample_rate = sample_rate
        self.start_idxs = []
        self.end_idxs = []
        self.name = name
        for sample in samples:
            start_idx, end_idx = sample.start_idx, sample.end_idx
            self.start_idxs.append(start_idx)
            self.end_idxs.append(end_idx)
            self.interval.add((start_idx, end_idx, sample))
    
    def __len__(self):
        return len(self.waveform)
        
    def num_frames(self, frame_length):
        return len(self) // frame_length
    
    def resample(self, new_sample_rate):
        """ Resamples ``self.waveform`` to specified sampling rate. """
        data = self.waveform
        number_of_samples = round(len(data) * float(new_sample_rate) / self.sample_rate)
        self.waveform = sps.resample(data, number_of_samples)
        self.sample_rate = new_sample_rate
    
    @methodtools.lru_cache()
    def get_frequencies_from_offset(self, interval_start_idx, interval_end_idx, min_presence=0.5):
        """ Returns the frequencies present between index ``start_idx`` and ``end_idx``. """
        # Find samples in scaled interval
        frame_length = interval_end_idx - interval_start_idx
        scaled_interval_start = math.floor(self.original_sample_rate / self.sample_rate * interval_start_idx)
        scaled_interval_end = math.ceil(self.original_sample_rate / self.sample_rate * interval_end_idx)
        samples_in_interval = list(
            self.interval.find((scaled_interval_start + 1e-8, scaled_interval_end - 1e-8))
        )
        samples = []
        for start_idx, end_idx, sample in samples_in_interval:
            span = min(end_idx, scaled_interval_end) - max(start_idx, scaled_interval_start)
            if span / frame_length >= min_presence:
                samples.append(sample)
        if len(samples):
            merged_chunk = AnnotatedAudioChunk.merge_samples(samples)
            return merged_chunk.frequencies
        else:
            return []
        
        
class GuitarDataLoader:
    """ Loads audio data. 
    
        - ``sample_rate`` is in Hz
        - ``sample_length`` is in # samples
        - ``dataset`` is some of: ['idmt', 'nsynth', 'guitarset']
        
        - ``batch_by_track`` (bool): If true, returns lists of lists, where
            sublists contain waveforms within a track. Useful for LSTM-training,
            where tracks must come in sequential order. Otherwise, returns
            everything in a big list.
        - ``val_split`` (float, default=0.10): size of validation set to return
    """
    def __init__(self, sample_rate, string_based=False, 
            datasets=['idmt'], normalize_audio=True, 
            skip_empty_samples=False,
            batch_by_track=False, val_split=0.10,
            shuffle_tracks=True, max_polyphony=None
        ):
        self.sample_rate = sample_rate
        self.shuffle_tracks = shuffle_tracks
        self.skip_empty_samples = skip_empty_samples
        self.batch_by_track = batch_by_track
        self.val_split = val_split
        self.max_polyphony = max_polyphony
        
        assert not self.skip_empty_samples or not self.batch_by_track, "should not train LSTMS without including empty samples for context"
        
        if not set(datasets).issubset(set(dataset_load_funcs.keys())):
            raise ValueError(f"Can't load datasets {datasets}")
        self.datasets = datasets
    
    def get_tracks(self):
        tracks = []
        for dataset in self.datasets:
            print(f'--> GuitarDataLoader loading dataset {dataset}')
            if dataset not in dataset_load_funcs:
                raise ValueError(f'Unknown dataset {dataset}')
            tracks += dataset_load_funcs[dataset]()
        return tracks
    
    def load(self):
        """ Loads dataset and returns numpy array (X, y). Returns audio in chunks
        of ``self.sample_length`` sampled at ``self.sample_rate`` Hz.
        
        If ``batch_by_track`` is True, returns list of tracks where each track 
        is an (x, y) tuple representing its waveforms of length 
        ``self.sample_length`` and respective annotations. Having chunks in
        track-order is useful when learning one chunk depends on the previous,
        like with LSTMs.
        """
        tracks = self.get_tracks()
        
        for track in tqdm.tqdm(tracks, desc='Resampling tracks'):
            # TODO speed up with multiprocessing?
            track.resample(self.sample_rate)
        
        if self.shuffle_tracks: 
            random.shuffle(tracks)
        
        # Split into train and val data
        # (doing this perfectly is a DP problem, but we can approximate
        # the solution effectively using a greedy algorithm, especially if
        # our training data is plentiful and track size is somewhat uniform)
        train_tracks, val_tracks = [], []
        total_len_tracks = sum((len(track) for track in tracks))
        expected_len_val_tracks = round(total_len_tracks * self.val_split)
        
        len_tracks_seen = 0
        len_val_tracks = 0
        for track in tracks:
            if len_val_tracks < expected_len_val_tracks: 
                val_tracks.append(track)
                len_val_tracks += len(track)
            else:
                train_tracks.append(track)
            len_tracks_seen += len(track)
        
        # Flatten out, if not batch by track
        print(f'Observed val split {len_val_tracks / total_len_tracks} for desired val split {self.val_split:.2f}')
        print(f'GuitarDataLoader loaded {total_len_tracks / self.sample_rate:.2f}s worth of audio')
        
        return train_tracks, val_tracks

def load_idmt_tiny():
    return load_idmt(perc=0.05)

def load_idmt(perc=None, root_dir='/p/qdata/jm8wx/other/audio/data/IDMT-SMT-GUITAR_V2/'):
    """ Loads the IDMT-ST dataset. Returns List<Track>. """
    import xml.etree.ElementTree as ElementTree
    # TODO support datasets 3,4
    # wav_files = sorted(glob.glob(os.path.join(root_dir, 'dataset1/*/audio/*')) + glob.glob(os.path.join(root_dir, 'dataset2/audio/*')))
    wav_files = sorted(glob.glob(os.path.join(root_dir, 'dataset*/**/audio/*.wav'), recursive=True))
    # xml_files = sorted(glob.glob(os.path.join(root_dir, 'dataset1/*/annotation/*')) + glob.glob(os.path.join(root_dir, 'dataset2/annotation/*')))
    xml_files = sorted(glob.glob(os.path.join(root_dir, 'dataset*/**/annotation/*.xml'), recursive=True))
    
    wav_files = [f for f in wav_files if 'dataset4' not in f] # TODO: Process dataset4 audio from csv
    xml_files = [f for f in xml_files if 'dataset4' not in f]
    if not wav_files: 
        raise FileNotFoundError('no audio files found!')
    
    if perc:
        new_num_samples = int(len(wav_files) * perc)
        wav_files = wav_files[:new_num_samples]
        xml_files = xml_files[:new_num_samples]
    
    tracks = []
    total_length = 0
    total_audio_s = 0
    for wav_file, xml_file in zip(wav_files, xml_files):
        root = ElementTree.parse(xml_file).getroot()
        # load audio file 
        raw_waveform, sample_rate = torchaudio.load(wav_file)  # load tensor from file
        raw_waveform = raw_waveform.squeeze().numpy()
        total_audio_s += len(raw_waveform) / sample_rate
        # load annotations
        num_annotations = len(root.findall('transcription/event/pitch'))
        samples = []
        for i in range(num_annotations):
            # obj['audio_file_name'] = wav_file
            midi = int(root.findall('transcription/event/pitch')[i].text)
            pitch = midi_to_hz(midi)
            onset = float(root.findall('transcription/event/onsetSec')[i].text)
            offset = float(root.findall('transcription/event/offsetSec')[i].text)
            fret_number = int(root.findall('transcription/event/fretNumber')[i].text)
            string_number = int(root.findall('transcription/event/stringNumber')[i].text)
            # get relevant part of waveform
            start_sample_idx = int(onset * sample_rate)
            end_sample_idx = int(offset * sample_rate)
            # waveform_sample = raw_waveform[start_sample_idx:end_sample_idx]
            new_sample = AnnotatedAudioChunk(start_sample_idx, end_sample_idx, sample_rate, [pitch], [fret_number], [string_number])
            samples.append(new_sample)
        total_length += sum((s.length_in_seconds for s in samples))
        wav_file_short = wav_file
        if 'IDMT-SMT-GUITAR_V2/' in wav_file_short:
            # Shorten path from absolute system path to relative within dataset
            # -- this prevents really long absolute paths from clogging up
            # our data table
            wav_file_short = wav_file_short[wav_file_short.index('IDMT-SMT-GUITAR_V2/'):]
        track = Track('idmt', wav_file_short, samples, raw_waveform, sample_rate, name=xml_file)
        tracks.append(track)
    
    # Return samples
    print(f'IDMT Loaded {total_length:.2f}s of samples across all annotations ({total_audio_s:.2f}s of audio)')
    return tracks

def load_guitarset():
    """
    Loads GuitarSet from the mirdata package. Returns List<Track>.
    
    To download:
        >>> from mirdata import guitarset
        >>> guitarset.download() # takes a few minutes
    """
    tracks = []
    total_audio_s = 0
    for track_id in tqdm.tqdm(guitarset.DATA.index.keys(), desc='Loading GuitarSet'):
        track = guitarset.Track(track_id)
        waveform, sample_rate = torchaudio.load(track.audio_mic_path)
        waveform = waveform.flatten()
        total_audio_s += len(waveform) / sample_rate
        samples = []
        for string_name, f0data in track.notes.items():
            string_number = 6 - (['E', 'A', 'D', 'G', 'B', 'e'].index(string_name))
            for start, end, midi in zip(f0data.start_times, f0data.end_times, f0data.notes):
                string_numbers = [string_number]
                freq = midi_to_hz(midi)
                start_idx = int(start * sample_rate)
                end_idx = int(end * sample_rate)
                fret_numbers = [0] # TODO where in this data are the fret numbers?? Are they anywhere or must I infer them here?
                this_waveform = waveform[start_idx : end_idx]
                chunk = AnnotatedAudioChunk(
                    start_idx, end_idx, sample_rate, [freq], fret_numbers, string_numbers,
                )
                samples.append(chunk)
        track = Track('guitarset', track_id, samples, waveform, sample_rate, name=track_id)
        tracks.append(track)
    print(f'GuitarSet loaded {total_audio_s:.2f}s of audio ({len(tracks)} tracks)')
    return tracks

def load_nsynth(use_guitar=True, use_bass=False, use_acoustic_guitar=False, use_full_dataset=False, perc=None):
    # TODO use multiprocessing for this and other data loaders.
    # TODO or load nsynth from tensorflow_datasets?
    json_files = [f'/p/qdata/jm8wx/other/audio/data/NSynth/nsynth-{folder}/examples.json' for folder in ('train', 'test')]
    # Bad samples
    bad_nsynth_insts = {
        'guitar_acoustic_023',
        'guitar_acoustic_025',
        'guitar_acoustic_031',
        'guitar_acoustic_036',
    }
    
    tracks = []
    total_length = 0
    file_prefixes = collections.Counter()
    for file in json_files:
        data = json.loads(open(file).read())
        folder = os.path.join(os.path.dirname(file))
        for key in data.keys():
            prefix = key[:key.find('_')]
            file_prefixes[prefix] += 1
        # data types: {'mallet', 'reed', 'string', 'brass', 'guitar', 'bass', 'vocal', 'flute', 'synth', 'organ', 'keyboard'}
        sample_names = [sample_name for sample_name in sorted(data.keys()) 
            if (('guitar_acoustic' in sample_name) and use_acoustic_guitar)
            or (('guitar' in sample_name) and (use_guitar or use_full_dataset)) 
            or (('bass' in sample_name) and (use_bass or use_full_dataset))
            or (('string' in sample_name or 'synth' in sample_name or 'keyboard' in sample_name) and use_full_dataset)
        ]
        if perc:
            assert 0 < perc <= 1, "percentage of NSynth to use must be on (0, 1]"
            sample_names = sample_names[:int(perc * len(sample_names))]
        for sample_name in tqdm.tqdm(sample_names, desc=f'Loading NSynth data from {file}'):
            if sample_name in bad_nsynth_insts:
                print(f'Skipping bad sample {sample_name}')
                continue
            wav_file = os.path.join(folder, 'audio', f'{sample_name}.wav')
            raw_waveform, sample_rate = torchaudio.load(wav_file)
            raw_waveform = raw_waveform.flatten()
            # Trim leading and trailing silence.
            if use_acoustic_guitar:
                raw_waveform, trimmed_region = librosa.effects.trim(raw_waveform, top_db=20)
            # TODO Fake string and fret number?
            sample_json = data[sample_name]
            midi = sample_json['pitch']
            # sample_json: {
            #   'qualities': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            #   'pitch': 66, 'note': 72288, 
            #   'instrument_source_str': 'acoustic', 
            #   'velocity': 100, 'instrument_str': 'guitar_acoustic_010', 
            #   'instrument': 219, 'sample_rate': 16000, 
            #   'qualities_str': [], 'instrument_source': 0, 
            #   'note_str': 'guitar_acoustic_010-066-100', 
            #   'instrument_family': 3, 
            #   'instrument_family_str': 'guitar'
            #   }
            pitch = midi_to_hz(midi)
            new_sample = AnnotatedAudioChunk(
                0, len(raw_waveform), 
                sample_rate, 
                [pitch], [0], [1]
            )
            total_length += new_sample.length_in_seconds
            track = Track('nsynth', sample_name, [new_sample], raw_waveform, sample_rate, name=sample_name)
            tracks.append(track)
    print(f'NSynth file counts by prefix: {file_prefixes}')
    print(f'NSynth loaded {total_length:.2f}s of audio ({len(tracks)} tracks)')
    return tracks

def load_nsynth_full():
    return load_nsynth(use_full_dataset=True)

def load_nsynth_piano():
    raise NotImplementedError()

def load_nsynth_acoustic_guitar():
    return load_nsynth(use_guitar=False, use_bass=False, use_acoustic_guitar=True, use_full_dataset=False)

dataset_load_funcs = { 
    'guitarset': load_guitarset, 
    'idmt': load_idmt,
    'idmt_tiny': load_idmt_tiny,
    'nsynth': load_nsynth,
    'nsynth_full': load_nsynth_full,
    'nsynth_acoustic_guitar': load_nsynth_acoustic_guitar
    'nsynth_piano': load_nsynth_piano,
}


if __name__ == '__main__':
    # print('debug run - loading guitarset for profiling')
    # load_guitarset()
    print('debug run - loading NSynth acoustic guitar')
    load_nsynth_acoustic_guitar()
