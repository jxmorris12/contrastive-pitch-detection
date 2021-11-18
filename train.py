import argparse
import json
import os
import pathlib
import random
import re
import time
import torch
import wandb

import numpy as np

from callbacks import LogNoteEmbeddingStatisticsCallback, LogRecordingSpectrogramCallback, VisualizePredictionsCallback
from dataloader import MusicDataLoader, dataset_load_funcs
from generator import AudioDataGenerator
from models import ContrastiveModel, CREPE
from metrics import (
    NStringChordAccuracy, pitch_number_acc,
    precision, recall, f1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ask the OS how many cpus we have (stackoverflow.com/questions/1006289)
num_cpus = len(os.sched_getaffinity(0))
def set_random_seed(r):
    random.seed(r)
    np.random.seed(r)
    torch.manual_seed(r)
    torch.cuda.manual_seed(r)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for adam optimizer')
    parser.add_argument('--sample_rate', type=int, default=16_000, help='audio will be resampled to this sample rate before being passed to the model (measured in Hz)')
    parser.add_argument('--frame_length', '--frame_length', type=int, default=1024, help='length of audio samples (in number of datapoints)')

    parser.add_argument('--val_split', type=float, default=0.05, help='Size of the validation set relative to total number of waveforms. Will be an approximate, since individual tracks are grouped within train or validation only.')
    parser.add_argument('--randomize_train_frame_offsets', '--rtfo', type=bool, default=True, 
        help="Whether to add some randomness to frame offsets for training data")
    parser.add_argument('--contrastive', default=False, 
        action='store_true', help='train with contrastive loss')
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
    
    return args

def get_model(args):
    if args.contrastive:
        crepe = CREPE(model='tiny', num_output_nodes=88, out_activation=None)
        return ContrastiveModel(crepe, args.min_midi, args.max_midi, args.embedding_dim)
    else:
        return CREPE(model='tiny', num_output_nodes=88, out_activation='sigmoid')
    
def main():
    args = parse_args()

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
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    
    print('len(train_generator):', len(train_generator))
    print('len(val_generator):', len(val_generator))
    
    model = get_model(args).to(device)
    print(model)

    print('training with optimizer Adam')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pitch_number_acc': pitch_number_acc,
        'n_string_acc_multi': NStringChordAccuracy('multi')
    }
    # metrics = ['categorical_accuracy', pitch_number_acc]
    for n_strings in range(1, 7):
        if n_strings > args.max_polyphony:
            # Don't show metrics for chords we're not training on
            break
        metrics[f'n_string_acc_{n_strings}'] = NStringChordAccuracy(n_strings)

    wandb.watch(model)

    #
    # callbacks
    #
    time_str = time.strftime("%Y%m%d-%H%M%S")
    model_folder = os.path.join('outputs', f'crepe-{time_str}')
    pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)
    print(f'Saving args & model to {model_folder}')
    with open(os.path.join(model_folder, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    # TODO(jxm): implement saving best models with pytorch
    
    callbacks = []
    # TODO(jxm): reinstate this callback with a piano piece
    # callbacks.append(LogRecordingSpectrogramCallback(args))
    callbacks.append(VisualizePredictionsCallback(args, model, val_generator, validation_steps))
    if args.contrastive:
        callbacks.append(LogNoteEmbeddingStatisticsCallback(model))
    
    log_train_metrics_interval = int(steps_per_epoch / 10.0)
    print(f'Total num steps = ({steps_per_epoch} steps_per_epoch) * ({args.epochs} epochs) = {steps_per_epoch * args.epochs} ')
    total_num_steps = steps_per_epoch * args.epochs
    for step in range(total_num_steps):
        # Pre-epoch callbacks.
        epoch = int(step / steps_per_epoch)
        if step % steps_per_epoch == 0:
            for callback in callbacks:
                callback.on_epoch_begin(epoch, step)
        # Get data and predictions.
        (data, labels) = train_generator[step % len(train_generator)]
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Compute loss and backpropagate.
        if args.contrastive:
            loss = model.contrastive_loss(output, labels)
            output = model.get_probs(output) # Get actual probabilities for notes (for logging)
        else:
            loss = torch.nn.functional.binary_cross_entropy(output, labels)
        wandb.log({ 'train_loss': loss }, step=step)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Compute train metrics.
        # TODO(jxm): Mechanism for averaging metrics instead of logging just for one batch (too noisy).
        if (step+1) % log_train_metrics_interval == 0:
            logging.info('*** Computing training metrics for epoch %d (step %d) ***', epoch, step)
            for name, metric in metrics.items():
                metric_name = f'train_{name}'
                metric_val = metric(output, labels)
                wandb.log({ metric_name: metric_val }, step=step)
        # Post-epoch callbacks.
        if (step+1) % steps_per_epoch == 0:
            # Compute validation metrics.
            # TODO(jxm): avg validation metrics?
            logging.info('*** Computing validation metrics for epoch %d (step %d) ***', epoch, step)
            for batch in val_generator:
                (data, labels) = batch
                data, labels = data.to(device), labels.to(device)
                with torch.no_grad():
                    output = model(data)
                    if args.contrastive:
                        loss = model.contrastive_loss(output, labels)
                        output = model.get_probs(output) # Get actual probabilities for notes (for logging)
                    else:
                        loss = torch.nn.functional.binary_cross_entropy(output, labels)
                wandb.log({ 'val_loss': loss }, step=step)
                for name, metric in metrics.items():
                    metric_name = f'val_{name}'
                    metric_val = metric(output, labels)
                    logging.info('\t%s = %f', metric_name, metric_val)
                    wandb.log({ metric_name: metric_val}, step=step)
            # Also shuffle training data after each epoch.
            train_data_loader.on_epoch_end()
    
    print(f'training done! model saved to {model_folder}')

if __name__ == '__main__': 
    main()


