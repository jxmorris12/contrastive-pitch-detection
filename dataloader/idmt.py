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


