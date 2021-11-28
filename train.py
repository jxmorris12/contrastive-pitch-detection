import argparse
import json
import logging
import os
import pathlib
import random
import re
import time
import torch
import tqdm
import wandb

import numpy as np

from callbacks import LogNoteEmbeddingStatisticsCallback, LogRecordingSpectrogramCallback, VisualizePredictionsCallback
from dataloader import MusicDataLoader, dataset_load_funcs
from generator import AudioDataGenerator
from models import ContrastiveModel, CREPE, Bytedance_Regress_pedal_Notes, S4Model
from metrics import (
    categorical_accuracy, pitch_number_acc, NStringChordAccuracy,
    precision, recall, f1
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_cpus = len(os.sched_getaffinity(0)) # ask the OS how many cpus we have (stackoverflow.com/questions/1006289)

WANDB_ENABLED = wandb.setup().settings.mode != "disabled"

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
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for adam optimizer')
    parser.add_argument('--sample_rate', type=int, default=16_000, help='audio will be resampled to this sample rate before being passed to the model (measured in Hz)')
    parser.add_argument('--frame_length', '--frame_length', type=int, default=1024, help='length of audio samples (in number of datapoints)')
    parser.add_argument('--num_fake_nsynth_chords', type=int, default=0,
        help='number of fake NSynth chord tracks to include. Will over-write train set!')
    parser.add_argument('--model', type=str, default='bytedance_tiny', help='model to use for training',
        choices=('crepe_tiny', 'crepe_full', 'bytedance', 'bytedance_tiny', 's4'))

    parser.add_argument('--randomize_val_and_training_data', '--rvatd', default=False,
        action='store_true', help='shuffle validation and training data')
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
    
    assert args.min_midi < args.max_midi, "Must provide a positive range of MIDI values where max_midi > min_midi"
    set_random_seed(args.random_seed)
    
    return args

def get_model(args):
    # TODO(jxm): support nn.DataParallel here
    num_output_nodes = 256 if args.contrastive else 88
    out_activation = 'softmax' if args.max_polyphony == 1 else 'sigmoid'
    if args.model == 'bytedance':
        model = Bytedance_Regress_pedal_Notes(
            num_output_nodes, out_activation, tiny=False
        )
    elif args.model == 'bytedance_tiny':
        model = Bytedance_Regress_pedal_Notes(
            num_output_nodes, out_activation, tiny=True
        )
    elif args.model == 's4':
        model = S4Model(
            d_output=num_output_nodes,
            out_activation=out_activation
        )
    elif args.model == 'crepe_tiny':
        model = CREPE(
            model='tiny',
            num_output_nodes=num_output_nodes,
            load_pretrained=False,
            out_activation=out_activation
        )
    elif args.model == 'crepe_full':
        model = CREPE(
            model='full',
            num_output_nodes=num_output_nodes,
            load_pretrained=False,
            out_activation=out_activation
        )
    else:
        raise ValueError(f'Invalid model {args.model}')

    if args.contrastive:
        return ContrastiveModel(model, args.min_midi, args.max_midi, num_output_nodes)
    else:
        return model
    
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
    if args.num_fake_nsynth_chords == 0:
        train_data_loader = MusicDataLoader(sample_rate, frame_length, 
            datasets=['nsynth_chords_train'],
            batch_by_track=False, val_split=0.0
        )
        train_tracks = train_data_loader.load()
    else:
        # Don't load train tracks if we're randomly generating!
        train_tracks = []

    val_data_loader = MusicDataLoader(sample_rate, frame_length, 
        datasets=['nsynth_chords_valid', 'nsynth_chords_test'],
        batch_by_track=False, val_split=0.0
    )
    val_tracks = val_data_loader.load()

    if args.randomize_val_and_training_data:
        print('Shuffling val and train tracks')
        num_val_tracks = len(val_tracks)
        all_tracks = train_tracks + val_tracks
        random.shuffle(all_tracks)
        val_tracks = all_tracks[:num_val_tracks]
        train_tracks = all_tracks[num_val_tracks:]
    
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
        num_fake_nsynth_chords=args.num_fake_nsynth_chords,
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    metrics = {
        'categorical_accuracy': categorical_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pitch_number_acc': pitch_number_acc,
        'n_string_acc_multi': NStringChordAccuracy('multi')
    }
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
    if WANDB_ENABLED:
        # only compute this stuff if w&b is not disabled
        callbacks.append(VisualizePredictionsCallback(args, model, val_generator, validation_steps, str_prefix='val_'))
        # TODO(jxm): Reuse train predictions instead of recomputing them
        callbacks.append(VisualizePredictionsCallback(args, model, train_generator, validation_steps, str_prefix='train_'))
        if args.contrastive:
            callbacks.append(LogNoteEmbeddingStatisticsCallback(model))
    
    print(f'Total num steps = ({steps_per_epoch} steps_per_epoch) * ({args.epochs} epochs) = {steps_per_epoch * args.epochs} ')
    total_num_steps = steps_per_epoch * args.epochs
    log_interval = int(steps_per_epoch / 10.0) # TODO(jxm): argparse for logs_per_epoch?
    pbar = tqdm.trange(total_num_steps)
    for step in pbar:
        # Pre-epoch callbacks.
        epoch = int(step / steps_per_epoch)
        pbar.set_description(f'Training (Epoch {epoch})')
        if step % steps_per_epoch == 0:
            # Callbacks.
            for callback in callbacks:
                callback.on_epoch_begin(epoch, step)
            # Adjust learning rate.
            if epoch > 0: scheduler.step()
            # Save model to disk.
            if (epoch+1) % 10 == 0:
                checkpoint = {
                    'step': step, 
                    'model': model.state_dict()
                }
                checkpoint_path = os.path.join(
                    model_folder, f'{epoch}_epochs.pth')   
                torch.save(checkpoint, checkpoint_path)
                logging.info(f'Model saved to {checkpoint_path}')
        # Get data and predictions.
        (data, labels) = train_generator[step % len(train_generator)]
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        # Compute loss and backpropagate.
        if args.contrastive:
            loss = model.contrastive_loss(output, labels)
            output = model.get_probs(output) # Get actual probabilities for notes (for logging)
        else:
            loss = torch.nn.functional.binary_cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Post-epoch callbacks.
        if (step+1) % log_interval == 0:
            # Compute train metrics.
            # TODO(jxm): Mechanism for averaging metrics instead of logging just for one batch (too noisy).
            train_metrics_dict = { 
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
                'step': step, 'epoch': epoch 
            }
            logger.info('*** Computing training metrics for epoch %d (step %d) ***', epoch, step)
            for name, metric in metrics.items():
                metric_name = f'train_{name}'
                metric_val = metric(output, labels)
                train_metrics_dict[metric_name] = metric_val
                logging.info('\t%s = %f', metric_name, metric_val)
            wandb.log(train_metrics_dict)
            # Compute validation metrics.
            # TODO(jxm): avg validation metrics!
            logger.info('*** Computing validation metrics for epoch %d (step %d) ***', epoch, step)
            for batch in val_generator:
                (data, labels) = batch
                data, labels = data.to(device), labels.to(device)
                with torch.no_grad():
                    output = model(data)
                    if args.contrastive:
                        val_loss = model.contrastive_loss(output, labels)
                        output = model.get_probs(output) # Get actual probabilities for notes (for logging)
                    else:
                        val_loss = torch.nn.functional.binary_cross_entropy(output, labels)
                val_metrics_dict = { 'val_loss': val_loss.item(), 'step': step, 'epoch': epoch  }
                for name, metric in metrics.items():
                    metric_name = f'val_{name}'
                    metric_val = metric(output, labels)
                    val_metrics_dict[metric_name] = metric_val
                    logging.info('\t%s = %f', metric_name, metric_val)
                wandb.log(val_metrics_dict)
                break # TMP until we average val metrics!
            tqdm.tqdm.write(f'Train loss = {loss.item():.4f} / Val loss = {val_loss.item():.4f}')
            # Also shuffle training data after each epoch.
            train_generator.on_epoch_end()
    
    print(f'training done! model saved to {model_folder}')

if __name__ == '__main__': 
    main()


