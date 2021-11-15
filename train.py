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

from augmenter import Augmenter
from callbacks import LogRecordingSpectrogramCallback, VisualizePredictionsCallback
from dataloader import MusicDataLoader, dataset_load_funcs
from generator import AudioDataGenerator, AudioCombinationGenerator
from models import CREPE
from metrics import (
    MeanSquaredError, loss_1, loss_2, string_level_regression_loss, 
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
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for adam optimizer')
    parser.add_argument('--sample_rate', type=int, default=16_000, help='audio will be resampled to this sample rate before being passed to the model (measured in Hz)')
    parser.add_argument('--frame_length', '--frame_length', type=int, default=1024, help='length of audio samples (in number of datapoints)')
    parser.add_argument('--min_midi', type=int, default=40, help='minimum MIDI value to consider')
    parser.add_argument('--max_midi', type=int, default=84, help='maximum MIDI value to consider')
    parser.add_argument('--model', type=str, default='crepe-tiny-pretrained')
    parser.add_argument('--datasets', '--d', type=str, nargs='+', choices=dataset_load_funcs.keys(),
                default=['idmt', 'guitarset'], help='datasets to load')
    parser.add_argument('--val_split', type=float, default=0.05, help='Size of the validation set relative to total number of waveforms. Will be an approximate, since individual tracks are grouped within train or validation only.')
    parser.add_argument('--randomize_train_frame_offsets', '--rtfo', type=bool, default=True, 
        help="Whether to add some randomness to frame offsets for training data")
    
    # parser.add_argument('--augment_noise_chance', type=float, default=0.21, help='augmentation % chance noise')
    # parser.add_argument('--augment_max_noise_level', type=float, default=0.95,help='augmentation max noise level')
    # parser.add_argument('--augment_pitch_shift_chance', type=float, default=0.56, help='augmentation % chance pitch shift')
    # parser.add_argument('--augment_max_pitch_shift', type=float, default=0.25,help='augmentation max pitch shift (in semitones)')
    # parser.add_argument('--augment_zero_rate', type=float, default=0.02, help='augmentation % zero rate')
    # parser.add_argument('--augment_roll_chance', type=float, default=0.30, help='augmentation % roll chance')
    # parser.add_argument('--augment_combine', type=bool, default=False, help='combine multiple examples into a single augmented chord')
            
    parser.add_argument('--optimizer', help='optimizer for gradient descent',
        default='adam', choices=('sgd', 'adam'))

    parser.add_argument('--eager', '--run_eagerly', default=False, 
        action='store_true', help='run TensorFlow in eager execution mode')
    parser.add_argument('--tensorboard', '--use_tensorboard', default=False, 
        action='store_true', help='Save TensorBoard logs and profiling results')
    parser.add_argument('--quantize', default=False, 
        action='store_true', help='enable quantization aware training')
    parser.add_argument('--loss', help='loss function for training the model',
        default='classification', choices=('classification'))
    parser.add_argument('--lstm', default=0, type=int, choices=(0,1),
        help='add an LSTM layer' )
    parser.add_argument('--max_polyphony', type=int, default=float('inf'), choices=list(range(6)),
        help='If specified, will filter out frames with greater than this number of notes')

    parser.add_argument('--n_gpus', default=None, 
        type=int, help='distributes the model acros ``n_gpus`` GPUs')
    parser.add_argument('--normalize', default=False, 
        type=bool, help='normalize inputs before inputting them')
    parser.add_argument('--tf_debug', default=False, 
        action='store_true', help='enable tensorflow debugging features')
    
    args = parser.parse_args()
    assert args.lstm == False # not implemented in this codebase
    
    assert 0 <= args.val_split < 1, "val split must be on [0, 1)"
    
    assert args.min_midi < args.max_midi, "Must provide a positive range of MIDI values where max_midi > min_midi"
    set_random_seed(args.random_seed)
    
    if args.tf_debug:
        tf.debugging.set_log_device_placement(True)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Shut up tensorflow!
        
    if args.loss == 'reproduction':
        assert args.model == 'crepe-with-decoder', \
            'must use reproduction loss with decoder model'
    
    return args

def get_model(args):
    if args.loss == 'regression':
        num_output_nodes = 6
    else:
        num_output_nodes = (args.max_midi - args.min_midi + 1)

    #
    # Get base model based on ``args.model``.
    #
    crepe_re = r'^crepe-(tiny|small|medium|large|full)(?:-pretrained)?(?:-mixed-input)?$'
    crepe_match = re.search(crepe_re, args.model)
    if args.lstm and not crepe_match:
        raise NotImplementedError('LSTMs only implemented for CREPE (for now!)')
    
    if crepe_match:
        load_pretrained = '-pretrained' in args.model
        model_capacity = crepe_match.groups()[0]
        
        if load_pretrained and args.sample_rate != 16_000:
            raise RuntimeError(f'trying to load CREPE pretrained model with sample rate {args.sample_rate} != 16_000 Hz')
        
        model_cls = CREPE
        model = model_cls(
                model_capacity=model_capacity,
                load_pretrained=load_pretrained,
                input_dim=args.frame_length, 
                num_output_nodes=num_output_nodes,
                lstm=args.lstm,
            )
    else:
        raise ValueError(f'unknown model {args.model}')
    #
    # Quantize weights, if desired.
    #
    if args.quantize:
        raise ValueError('quantization-aware training not implemented - or at least, it does not work!')
        import tensorflow_model_optimization as tfmot
        model = tfmot.quantization.keras.quantize_model(model)
    
    if args.n_gpus:
        print(f'Distributing model across {args.n_gpus} GPUs')
        from tensorflow.python.keras.utils import multi_gpu_utils
        model = multi_gpu_utils.multi_gpu_model(model, gpus=args.n_gpus)
    
    return model
    
def main():
    tf.config.experimental_run_functions_eagerly(True)
    # TODO: Look into SpecAugment augmentation https://www.tensorflow.org/io/tutorials/audio#specaugment
    args = parse_args()
    # wandb_project_name = 'chord-detection' + ('-lstm' if args.lstm else '')
    wandb_project_name = 'crepe'
    wandb.init(entity='jxmorris12', project=os.environ.get('WANDB_PROJECT', wandb_project_name), job_type='train', config=args)
    args.run_name = wandb.run.id
    
    frame_length = args.frame_length
    sample_rate = args.sample_rate
    frame_length_in_ms = frame_length / sample_rate * 1000.0
    
    print(f'Frame length: {frame_length_in_ms:.2f}ms / {frame_length} points (sample_rate = {sample_rate} Hz)')
    assert len(args.datasets), "need data for training!"
    
    #
    # load data
    #
    label_format = 'categorical'
    
    data_loader = MusicDataLoader(sample_rate, frame_length, 
        datasets=args.datasets,
        batch_by_track=args.lstm,
        val_split=args.val_split,
        max_polyphony=args.max_polyphony,
    )
    train_tracks, val_tracks = data_loader.load()
    
    #
    # create data augmenter
    #
    # augmenter = Augmenter(
    #     noise_chance=args.augment_noise_chance, 
    #     max_noise_level=args.augment_max_noise_level,
    #     pitch_shift_chance=args.augment_pitch_shift_chance, 
    #     max_pitch_shift=args.augment_max_pitch_shift,
    #     zero_rate=args.augment_zero_rate, 
    #     roll_chance=args.augment_roll_chance,
    # )
    #
    # create data generator
    #
    
    # If model has 'mixed input' specified, it wants to take
    # two input types: the number of notes played and the
    # waveform itself
    include_num_notes_played = False
    train_generator = AudioDataGenerator(
        train_tracks, args.frame_length,
        args.max_polyphony,
        randomize_train_frame_offsets=args.randomize_train_frame_offsets,
        batch_size=args.batch_size, augmenter=augmenter,
        normalize_audio=args.normalize,
        label_format=label_format,
        min_midi=args.min_midi, max_midi=args.max_midi,
        sample_rate=args.sample_rate,
        batch_by_track=args.lstm,
        include_num_notes_played=False,
    )
    val_generator = AudioDataGenerator(
        val_tracks, args.frame_length,
        args.max_polyphony,
        randomize_train_frame_offsets=False,
        batch_size=args.batch_size, augmenter=None,
        normalize_audio=args.normalize,
        label_format=label_format,
        min_midi=args.min_midi, max_midi=args.max_midi,
        sample_rate=args.sample_rate,
        batch_by_track=args.lstm,
        include_num_notes_played=include_num_notes_played,
    )
    
    if args.augment_combine:
        print('Augmenting training data via chord combination')
        train_generator = AudioCombinationGenerator(train_generator)
    
    print('Wrapping datasets using tf.data.Dataset.from_generator...')
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    
    print('len(train_generator):', len(train_generator))
    print('len(val_generator):', len(val_generator))
    
    model = get_model(args)
    model.summary()
    #
    # model compile() and fit()
    #
    from tensorflow import keras # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
    if args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1)
    elif args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, clipnorm=1)
    else:
        raise ValueError(f'unknown optimizer {args.optimizer}')
    print(f'model.compile() with optimizer {args.optimizer}')
    
    metrics = ['categorical_accuracy', pitch_number_acc, NStringChordAccuracy('multi')]
    
    for n_strings in range(1, 7):
        if n_strings > args.max_polyphony:
            # Don't show metrics for chords we're not training on
            break
        metrics.append(NStringChordAccuracy(n_strings))

    metrics += [tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score]
    
    loss_weights = None
    if args.loss == 'classification':
        raise NotImplementedError()
        # loss_fn = ?
    else:
        raise ValueError(f'Unrecognized loss function {loss_fn}')
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics,
        run_eagerly=args.eager, loss_weights=loss_weights,)
    #
    # callbacks
    #
    time_str = time.strftime("%Y%m%d-%H%M%S")
    model_folder = os.path.join('outputs', f'{args.model}-{time_str}')
    pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)
    print(f'Saving args & model to {model_folder}')
    with open(os.path.join(model_folder, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    # model_path_format = os.path.join(model_folder, 'weights.{epoch:02d}-.hdf5')
    best_model_path_format = os.path.join(model_folder, 'weights.best.{epoch:02d}.hdf5')

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=75, verbose=1)
    save_best_model = keras.callbacks.ModelCheckpoint(best_model_path_format, save_best_only=True, monitor='val_loss')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.316227766, patience=2, min_lr=1e-10, verbose=1)
    
    callbacks = [early_stopping, save_best_model, reduce_lr, WandbCallback()]
    
    if 'reproduction' not in args.loss:
        if (not 'mixed-input' in args.model) and (not 'regression' in args.loss):
            # This callback only works for models that take a single waveform input (for now)
            callbacks.append(LogRecordingSpectrogramCallback(args))
        #callbacks.append(VisualizePredictionsCallback(args, val_generator, validation_steps)) ## TODO(jxm): figure out why this doesn't work!
    
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


