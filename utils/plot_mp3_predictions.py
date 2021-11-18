#
# jm8wx 10/25/20
#

# TODO(jxm): update to work with torch models [11/18/2021]

# model imports
import models

# program imports
# from keras.models import load_model
import matplotlib.pyplot as plt
from natsort import natsorted # ['11', 1', '10'] -> ['1', '10', '11']
from scipy import signal

import glob
import json
import numpy as np
import os
import seaborn as sns
# import tensorflow as tf
import torchaudio
import tqdm
import utils


## fix bug -- stackoverflow.com/questions/50204556
import matplotlib
matplotlib.use('Agg')

# --> TODO argparse these things
# model_folder = './outputs/crepe-tiny-pretrained-20201026-110556'
model_folder = './outputs/crepe-with-decoder-20210216-214609/'
audio_file = 'day_tripper_intro.mp3'
# audio_file = 'stairway_to_heaven.mp3' # in /samples/audio

def preprocess(arr):
    return (arr - arr.mean()) / arr.std()

def get_model_predictions(model, arr, sample_length, bins_per_window, min_midi, max_midi, batch_size):
    """ Gets model predictions for input array ``arr``. """
    hop_size = round(sample_length / bins_per_window)
    samples = []
    for i in tqdm.trange(0, len(arr), hop_size, desc='getting model predictions'):
        if i + sample_length > len(arr):
            break
        wav_sample = arr[i:i+sample_length]
        wav_sample = preprocess(wav_sample)
        samples.append(wav_sample)
    samples = np.vstack(samples)
    tensor_input = tf.convert_to_tensor(samples)
    preds = model.predict(tensor_input, batch_size=batch_size)
    if isinstance(preds, tuple):
        # Assume two outputs (autoencoder). Take second one.
        preds = preds[1]
    preds = preds.T # transpose -> axes: [midi, time]
    assert preds.shape[0] == (max_midi - min_midi + 1)
    return preds

def predict_audio_file(model, wav, sample_rate, sample_length, min_midi, max_midi, 
    bins_per_window=4, batch_size=32, return_image=False):
    """ Runs predictions using ``model`` on ``wav`` and generates a spectrogram.
    
    If ``return_image`` is true, returns the image object. otherwise, writes
    to file. 
    """
    hop_size = round(sample_length / bins_per_window)
    # get model predictions
    preds = get_model_predictions(model, wav, sample_length, bins_per_window, min_midi, max_midi, batch_size)
    # create & format plot
    fig_scale = max((len(wav) / sample_rate) / 16.0, 1.0)
    w = 50 * fig_scale
    w = min(w, 20_000)
    h = w / 3.5
    w, h = round(w), round(h)
    plt.figure(figsize=(w,h))
    ax = sns.heatmap(preds, cmap='viridis')
    x_time_vals = [int(t.get_text()) for t in plt.xticks()[1]]
    x_time_vals = [(t * hop_size) / sample_rate for t in x_time_vals]
    x_time_vals = [f'{t:.2f}' for t in x_time_vals]
    ax.set_xticklabels(x_time_vals)
    y_midi_vals = [int(t.get_text()) + min_midi for t in plt.yticks()[1]]
    y_midi_vals = map(str, y_midi_vals)
    ax.set_yticklabels(y_midi_vals)
    ax.set(xlabel='time', ylabel='MIDI', title=audio_file)
    # save plot
    plt.tight_layout()
    
    if return_image:
        image = utils.pyplot_to_numpy(plt)
        return image
    else:
        outfile = os.path.join('samples', 'plots', (audio_file + '.png'))
        plt.savefig(outfile)
        plt.cla()
        plt.close()
        print('saved figure to:', outfile)

def load_wav(wav_path, desired_sample_rate):
    wav, file_sr = torchaudio.load(wav_path)
    if wav.ndim > 1:
        wav = wav.sum(dim=0) # sum to one channel
    if file_sr != desired_sample_rate:
        new_num_samples = round(len(wav) * desired_sample_rate / file_sr)
        print(f'resampling wav from length {len(wav)} to {new_num_samples}')
        wav = signal.resample(wav, new_num_samples)
    return wav

def main():
    # load model
    model_paths = glob.glob(os.path.join(model_folder, 'weights.best*.hdf5'))
    model_path = natsorted(model_paths)[-1] # Take the last best model
    print('loading model:', model_path)
    # load args
    args_path = os.path.join(model_folder, 'args.json')
    args = json.load(open(args_path, 'r'))
    sample_rate = args['sample_rate']
    # TODO update custom objects / load models better
    try:
        model = load_model(model_path)
    except ValueError:
        model = models.CREPE(
            'tiny',
            hidden_representation_size=64,
            frame_length=args['frame_length'],
            num_output_nodes=(args['max_midi']-args['min_midi']+1)
        )
        model.load_weights(model_path)
    # load sound file
    wav_path = os.path.join('samples', 'audio', audio_file)
    wav = load_wav(wav_path, sample_rate)
    # predict
    predict_audio_file(model, wav, sample_rate, 
        args.get('sample_length', args['frame_length']), args['min_midi'], 
        args['max_midi'], return_image=False
    )

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Shut up tensorflow!
    sns.set_theme() # set nice seaborn theme (seaborn.pydata.org/generated/seaborn.heatmap.html)
    main()