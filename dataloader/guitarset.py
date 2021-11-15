
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