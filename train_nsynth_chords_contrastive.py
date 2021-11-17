import argparse
import json
import os
import pathlib
import random
import re
import time
import wandb

import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback # https://docs.wandb.com/library/integrations/keras

from callbacks import LogNoteEmbeddingStatisticsCallback, LogRecordingSpectrogramCallback, VisualizePredictionsCallback
from dataloader import MusicDataLoader, dataset_load_funcs
from generator import AudioDataGenerator
from models import CREPE, ContrastiveModel
from metrics import (
    MeanSquaredError, 
    NStringChordAccuracy, f1_score, pitch_number_acc
)

# ask the OS how many cpus we have (stackoverflow.com/questions/1006289)
num_cpus = len(os.sched_getaffinity(0))
def set_random_seed(r):
    random.seed(r)
    np.random.seed(r)
    tf.random.set_seed(r)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for adam optimizer')
    parser.add_argument('--sample_rate', type=int, default=16_000, help='audio will be resampled to this sample rate before being passed to the model (measured in Hz)')
    parser.add_argument('--frame_length', '--frame_length', type=int, default=1024, help='length of audio samples (in number of datapoints)')
    parser.add_argument('--embedding_dim', type=int, default=256, help='representation size of note embeddings')

    parser.add_argument('--val_split', type=float, default=0.05, help='Size of the validation set relative to total number of waveforms. Will be an approximate, since individual tracks are grouped within train or validation only.')
    parser.add_argument('--randomize_train_frame_offsets', '--rtfo', type=bool, default=True, 
        help="Whether to add some randomness to frame offsets for training data")
    parser.add_argument('--eager', '--run_eagerly', default=False, 
        action='store_true', help='run TensorFlow in eager execution mode')
    parser.add_argument('--max_polyphony', type=int, default=float('inf'), choices=list(range(6)),
        help='If specified, will filter out frames with greater than this number of notes')

    parser.add_argument('--n_gpus', default=None, 
        type=int, help='distributes the model acros ``n_gpus`` GPUs')
    
    args = parser.parse_args()
    args.min_midi = 21  # Bottom of piano
    args.max_midi = 108 # Top of piano
    
    assert 0 <= args.val_split < 1, "val split must be on [0, 1)"
    
    assert args.min_midi < args.max_midi, "Must provide a positive range of MIDI values where max_midi > min_midi"
    set_random_seed(args.random_seed)
    
    
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Shut up tensorflow!
    
    return args

def get_model(args):
    crepe = CREPE('medium', input_dim=args.frame_length, num_output_nodes=args.embedding_dim, load_pretrained=False,
        freeze_some_layers=False, add_intermediate_dense_layer=True,
        add_dense_output=True, out_activation=None)
    return ContrastiveModel(crepe, args.min_midi, args.max_midi, args.embedding_dim)

def main():
    args = parse_args()
    if args.eager: tf.config.run_functions_eagerly(True)

    wandb_project_name = 'nsynth_chords'
    wandb.init(entity='jxmorris12', project=os.environ.get('WANDB_PROJECT', wandb_project_name), job_type='train', config=args)
    args.run_name = wandb.run.id
    
    frame_length = args.frame_length
    sample_rate = args.sample_rate
    frame_length_in_ms = frame_length / sample_rate * 1000.0
    
    print(f'Frame length: {frame_length_in_ms:.2f}ms / {frame_length} points (sample_rate = {sample_rate} Hz)')
    
    #
    # load data
    train_data_loader = MusicDataLoader(sample_rate, frame_length, 
        datasets=['nsynth_chords_train'],
        batch_by_track=False, val_split=0.0
    )
    train_tracks = train_data_loader.load()

    val_data_loader = MusicDataLoader(sample_rate, frame_length, 
        datasets=['nsynth_chords_valid', 'nsynth_chords_test'],
        batch_by_track=False, val_split=0.0
    )
    val_tracks = val_data_loader.load()
    
    #
    # create data augmenter and generators
    #
    train_generator = AudioDataGenerator(
        train_tracks, args.frame_length,
        args.max_polyphony,
        randomize_train_frame_offsets=args.randomize_train_frame_offsets,
        batch_size=args.batch_size,
        augmenter=None,
        normalize_audio=False,
        label_format='categorical',
        min_midi=args.min_midi, max_midi=args.max_midi,
        sample_rate=args.sample_rate,
        batch_by_track=False,
    )
    val_generator = AudioDataGenerator(
        val_tracks, args.frame_length,
        args.max_polyphony,
        randomize_train_frame_offsets=False,
        batch_size=args.batch_size, augmenter=None,
        normalize_audio=False,
        label_format='categorical',
        min_midi=args.min_midi, max_midi=args.max_midi,
        sample_rate=args.sample_rate,
        batch_by_track=False,
    )

    # breakpoint()
    
    print('Wrapping datasets using tf.data.Dataset.from_generator...')
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    
    print('len(train_generator):', len(train_generator))
    print('len(val_generator):', len(val_generator))
    
    model = get_model(args)
    model.build([args.batch_size, args.frame_length])
    model.summary()
    #
    # model compile() and fit()
    #
    from tensorflow import keras # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1)
    print(f'model.compile() with optimizer Adam')
    
    # metrics = ['categorical_accuracy', pitch_number_acc, NStringChordAccuracy('multi')]
    metrics = ['categorical_accuracy', pitch_number_acc]
    metrics += [tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score]

    # Only apply metrics to label output.
    metrics = {
        'A_f': [],
        'probs': metrics
    }
    
    loss_fn = {
        'A_f': model.get_loss_fn(),
        # 'probs': None,
    }
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics,
        run_eagerly=args.eager)
    #
    # callbacks
    #
    time_str = time.strftime("%Y%m%d-%H%M%S")
    model_folder = os.path.join('outputs', f'crepe-{time_str}')
    pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)
    print(f'Saving args & model to {model_folder}')
    with open(os.path.join(model_folder, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    
    best_model_path_format = os.path.join(model_folder, 'weights.best.{epoch:02d}.h5')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=75, verbose=1)
    save_best_model = keras.callbacks.ModelCheckpoint(best_model_path_format, save_best_only=True, save_weights_only=True, monitor='val_loss')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.316227766, patience=2, min_lr=1e-10, verbose=1)
    
    callbacks = [early_stopping, save_best_model, reduce_lr, WandbCallback(), LogNoteEmbeddingStatisticsCallback(model)]
    
    # This callback only works for models that take a single waveform input (for now)
    # callbacks.append(LogRecordingSpectrogramCallback(args))
    callbacks.append(VisualizePredictionsCallback(args, val_generator, validation_steps))
    
    dataset_output_types = (float, float)
    train_generator = tf.data.Dataset.from_generator(
        train_generator._callable(args.epochs), output_types=dataset_output_types
    ).prefetch(tf.data.experimental.AUTOTUNE)
    val_generator = tf.data.Dataset.from_generator(
        val_generator._callable(args.epochs), output_types=dataset_output_types
    ).prefetch(tf.data.experimental.AUTOTUNE)
    num_cpus = len(os.sched_getaffinity(0))
        
        
    print('model.fit()')
    model.fit(
        x=train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=val_generator,
        use_multiprocessing=(num_cpus > 1),
        workers=num_cpus,
        epochs=args.epochs, 
        callbacks=callbacks,
    )
    
    print(f'training done! model saved to {model_folder}')

if __name__ == '__main__': 
    main()


