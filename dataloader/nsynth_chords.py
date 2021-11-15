import pickle

from .utils import AnnotatedAudioChunk, Track, midi_to_hz

SAMPLE_RATE = 16_000
TRAIN_PATH = '/home/jxm3/research/transcription/nsynth-chords/nsynth-keyboard-chords-train.p'
TEST_PATH = '/home/jxm3/research/transcription/nsynth-chords/nsynth-keyboard-chords-test.p'
VALID_PATH = '/home/jxm3/research/transcription/nsynth-chords/nsynth-keyboard-chords-valid.p'

def load_nsynth_chords(split='train', tiny=False):
    if split == 'train':
        raw_data = pickle.load(open(TRAIN_PATH, 'rb'))
    elif split == 'valid':
        raw_data = pickle.load(open(VALID_PATH, 'rb'))
    elif split == 'test':
        raw_data = pickle.load(open(TEST_PATH, 'rb'))
    else:
        raise ValueError(f'Unknown NSynth chords split {split}')

    data = []
    for item in raw_data:
        raw_waveform = item['audio']

        new_sample = AnnotatedAudioChunk(
            0, len(raw_waveform), 
            SAMPLE_RATE, 
            [midi_to_hz(n) for n in item['notes']], [0], [1]
        )

        sample_name = str(item['instrument_id']) + '_' + str(item['velocity']) + '_' + str(item['notes'])
        data.append(Track('nsynth', sample_name, [new_sample], raw_waveform, SAMPLE_RATE, name=sample_name))

    if tiny: data = data[:int(len(data) * 0.02)]
    
    return data

    