# tfds.audio.Nsynth

def load_nsynth(use_guitar=True, use_bass=False, use_keyboard=False, use_acoustic_guitar=False, use_full_dataset=False, perc=None):
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
            or (('keyboard' in sample_name) and (use_keyboard))
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

def load_nsynth_keyboard():
    return load_nsynth(use_keyboard=True)

def load_nsynth_acoustic_guitar():
    return load_nsynth(use_guitar=False, use_bass=False, use_acoustic_guitar=True, use_full_dataset=False)

dataset_load_funcs = { 
    'guitarset': load_guitarset, 
    'idmt': load_idmt,
    'idmt_tiny': load_idmt_tiny,
    'nsynth': load_nsynth,
    'nsynth_full': load_nsynth_full,
    'nsynth_acoustic_guitar': load_nsynth_acoustic_guitar
    'nsynth_keyboard': load_nsynth_keyboard,
}


if __name__ == '__main__':
    # print('debug run - loading guitarset for profiling')
    # load_guitarset()
    print('debug run - loading NSynth acoustic guitar')
    load_nsynth_acoustic_guitar()
