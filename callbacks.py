from typing import Dict

import abc
import collections
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import sklearn.manifold
import wandb
import tqdm

from utils import pyplot_to_numpy, get_prediction_type
from utils.plot_mp3_predictions import load_wav, predict_audio_file

import matplotlib # stackoverflow.com/questions/45993879
matplotlib.use('Agg') 

class Callback(abc.ABC):
    def on_epoch_begin(self, epoch: int, step: int):
        pass


class LogRecordingSpectrogramCallback(Callback):
    """ Generates a spectrogram based on model predictions for a given mp3 file.
    """
    def __init__(self, args, audio_file_path='samples/audio/day_tripper_intro.mp3',
        bins_per_window=4):
        self.frame_length = args.frame_length
        self.min_midi = args.min_midi
        self.max_midi = args.max_midi
        self.sample_rate = args.sample_rate
        self.frame_length = args.frame_length
        self.audio_file_path = audio_file_path
        self.wav = load_wav(audio_file_path, self.sample_rate)
        self.batch_size = args.batch_size
        self.bins_per_window = bins_per_window
    
    def get_spectrogram(self):
        return predict_audio_file(
            self.model, 
            self.wav, 
            self.sample_rate,
            self.frame_length,
            self.min_midi, 
            self.max_midi, 
            batch_size=self.batch_size,
            bins_per_window=self.bins_per_window, 
            return_image=True
        )
    
    def _log_spectrogram(self, epoch: int):
        image = self.get_spectrogram()
        name = f"spectrogram for {self.audio_file_path}"
        
        wandb.log({"spectrograms": [wandb.Image(image, caption=name)]}, step=epoch)
    
    def on_epoch_begin(self, epoch: int, step: int):
        """ At the beginning of each epoch, test model on chunks of the specified 
        file and log to TensorBoard.
        """
        self._log_spectrogram(epoch)
        

class VisualizePredictionsCallback(Callback):  
    """ Gets predictions from model using data from ``val_generator``, using
    ``num_val_batches`` of data. Runs every ``every_n_epochs`` epochs. Logs
    some instance-level and aggregated predictions to Weights & Biases.
    """
    def __init__(self, args, val_generator, num_validation_steps, every_n_epochs=3, num_validation_points=3*128):
        self.args = args
        self.val_generator = val_generator
        self.num_validation_steps = num_validation_steps
        self.every_n_epochs = every_n_epochs
        # Find number of batches necessary for 1024 validation predictions
        self.num_validation_points = num_validation_points
        self.num_val_batches = min(num_validation_steps, math.ceil( num_validation_points / args.batch_size))
        print(f'VisualizePredictionsCallback initialized with {self.num_val_batches} validation batches')
    
    def _log_overall_metrics(self, step, x, y_true, y_pred, frame_info):
        # Plot histogram of predictions
        hist_x = np.arange(self.args.min_midi, self.args.max_midi + 1)
        y_true_count = (y_true > 0.5).sum(axis=0)
        plt.figure(figsize=(12,3))
        sns.barplot(x=hist_x, y=y_true_count).set(title='y_true histogram')
        wandb.log({"y_true_hist": wandb.Image(plt)}, step=step)
        plt.cla()
        plt.close()
        y_pred_count = (y_pred > 0.5).sum(axis=0)
        plt.figure(figsize=(12,3))
        sns.barplot(x=hist_x, y=y_pred_count).set(title='y_pred histogram')
        wandb.log({"y_pred_hist": wandb.Image(plt)}, step=step)
        plt.cla()
        plt.close()
    
    def _plot_waveform(self, waveform):
        plt.figure(figsize=(4,2))
        sns.lineplot(data=waveform)
        return wandb.Image(pyplot_to_numpy(plt))
    
    def _plot_preds(self, preds):
        plt.figure(figsize=(8,2))
        x = np.arange(self.args.min_midi, self.args.max_midi + 1)
        sns.barplot(x=x, y=preds)
        return wandb.Image(pyplot_to_numpy(plt))
    
    def _plot_pred_types(self, step: int, pred_types: Dict[str, int]):
        """ Plots a bar graph of prediction types """
        data = [[pred_type, count] for (pred_type, count) in pred_types.items()]
        table = wandb.Table(data=data, columns = ["prediction_type", "count"])
        wandb.log({
            "val_pred_types" : 
            wandb.plot.bar(table, "prediction_type", "count", title="Validation prediction types")
        }, step=step)

    def _log_instance_level_metrics(self, step, x, y_true, y_pred, frame_info, max_instance_level_plots=128):
        table_columns = [
            'dataset', 'track', 'start_time', 'end_time', 
            'waveform', 'preds', 'pred_labels', 'true_labels', 
            'pred_type']
        table = wandb.Table(table_columns)
        pred_types = {}
        for pred_type in [
                'correct_silence', 'silence_instead_of_notes', 'notes_instead_of_silence',
                'correct_single_note', 'correct_chord', 'incorrect_single_note',
                'incorrect_chord', 'overpredicted_correct_single_note',
                'overpredicted_correct_chord', 'overpredicted_incorrect_single_note',
                'overpredicted_incorrect_chord', 'underpredicted_correct_chord',
                'underpredicted_incorrect_chord', 'other'
            ]: pred_types[pred_type] = 0
        # Print instance-level add metrics (like precision and accuracy)
        n = 0
        for waveform, true_labels, preds, frame_info in tqdm.tqdm(
                zip(x, y_true, y_pred, frame_info), 
                desc='VisualizePredictionsCallback plotting and logging instance-level metrics',
                total=len(frame_info)
            ):
            n += 1
            if (max_instance_level_plots is not None) and (n >= max_instance_level_plots):
                break
            row = [
                frame_info.dataset_name, frame_info.track_name, 
                frame_info.start_time, frame_info.end_time
            ]
            row.append(self._plot_waveform(waveform))
            row.append(self._plot_preds(preds))
            pred_midis = self.args.min_midi + preds.round().nonzero()[0]
            row.append(str(pred_midis))
            true_midis = self.args.min_midi + true_labels.nonzero()[0]
            row.append(str(true_midis))
            pred_type = get_prediction_type(pred_midis, true_midis)
            row.append(pred_type)
            pred_types[pred_type] += 1
            table.add_data(*row)
        
        self._plot_pred_types(step, pred_types)
        
        # Create an Artifact (versioned folder)
        artifact = wandb.Artifact(name="nsynth_chord_datasets", type="dataset")
        
        # .add the table to the artifact
        artifact.add(table, "val_predictions")
        
        # Finally, log the artifact
        wandb.log_artifact(artifact)
    
    def on_epoch_begin(self, epoch: int, step: int):
        # Have to re-get predictions because of this issue: 
        # https://github.com/keras-team/keras/issues/10472
        # TODO(jxm): Since we switched to pytorch, this isn't necessary. Optimize?
        rand_idxs = random.sample(range(self.num_validation_steps), self.num_val_batches)
        x, y_true, y_pred, frame_info = [], [], [], []
        for idx in rand_idxs:
            x_batch, y_true_batch, frame_info_batch = self.val_generator.__getitem__(idx, get_info=True)
            x_batch = x_batch[:self.num_validation_points]
            y_true_batch = y_true_batch[:self.num_validation_points]
            frame_info_batch = frame_info_batch[:self.num_validation_points]
            x.append(x_batch)
            y_true.append(y_true_batch)
            frame_info.extend(frame_info_batch)
            predictions = self.model.predict(x_batch)
            if isinstance(predictions, dict): # support the multi-output setting
                predictions = predictions['probs']
            y_pred.append(predictions)
        # Aggregate predictions from all batches
        x = np.vstack(x)
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        print(f'VisualizePredictionsCallback analyzing {len(frame_info)} predictions')
        self._log_overall_metrics(step, x, y_true, y_pred, frame_info)

        if (epoch + 1) % self.every_n_epochs > 0:
            print(f'Skipping VisualizePredictionsCallback instance-level predictions at epoch {epoch} / step {step}')
            return
        else:
            print(f'Logging VisualizePredictionsCallback instance-level predictions at epoch {epoch} / step {step}')
            self._log_instance_level_metrics(step, x, y_true, y_pred, frame_info)

class LogNoteEmbeddingStatisticsCallback(Callback):
    """ Plots some statistics related to a note embedding matrix.
    """
    def __init__(self, model):
        assert hasattr(model, 'embedding')
        self.embedding = model.embedding

    def _log_embedding_stats(self, step: int):
        """Log embedding norm, mean, and std."""
        emb_norm = torch.mean(torch.norm(self.embedding, p=2, axis=1))
        emb_std = torch.mean(torch.std(self.embedding, axis=0))
        wandb.log(
            {
                "note_embedding__norm": emb_norm,
                "note_embedding__std": emb_std,
            }, 
            step=step
        )

    def _plot_embedding_tsne(self, step: int):
        """Plot a 2D TSNE of all the embeddings, colored by note."""
        num_notes, emb_dim = self.embedding.shape
        # create the TSNE
        tsne = sklearn.manifold.TSNE(n_components=2, random_state=0, perplexity=5, learning_rate='auto', init='random')
        emb_dim_2 = tsne.fit_transform(self.embedding)
        # 12 colors, one per note
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#0e3f43', '#e0e0e0']
        emb_colors = []
        for i in range(num_notes):
            emb_colors.append(colors[i % len(colors)])
        # scatterplot of TSNE results
        plt.scatter(emb_dim_2[:,0], emb_dim_2[:,1], c=emb_colors)
        wandb.log({"note_embedding__tsne": wandb.Image(plt)}, step=step)
        plt.cla()
        plt.close()
        
    def _plot_embedding_similarities(self, step: int):
        """Plots all-to-all similarities between the embeddings."""
        with sns.axes_style("white"):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax = sns.heatmap(self.embedding @ self.embedding.T, square=True)
        wandb.log({"note_embedding__similarities": wandb.Image(plt)}, step=step)
        plt.cla()
        plt.close()
    
    def on_epoch_begin(self, epoch: int, step: int):
        """ At the beginning of each epoch, log embedding stats.
        """
        self._log_embedding_stats(step)
        self._plot_embedding_tsne(step)
        self._plot_embedding_similarities(step)
        