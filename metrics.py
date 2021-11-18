import abc
import torch
    
class Metric(abc.ABC):
    # TODO(jxm): Use this class to implement per-step averaging.
    @abc.abstractmethod
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        pass

def categorical_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """The number of multi-class predictions for y_true that are 100% correct (i.e. all classes
        in a given sample must be correctly predicted).
    """
    y_pred = torch.round(y_pred)
    all_correct = torch.all(y_true == y_pred, axis=1)
    return all_correct.sum() / len(all_correct)

def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Precision: (True Positives) / (True Positives + False Positives)"""
    y_pred = torch.round(y_pred)
    true_positives = torch.logical_and((y_true == 1), (y_pred == 1))
    false_positives = torch.logical_and((y_true == 0), (y_pred == 1))
    return true_positives / (true_positives + false_positives)

def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Recall: (True Positives) / (True Positives + False Negatives)"""
    y_pred = torch.round(y_pred)
    true_positives = torch.logical_and((y_true == 1), (y_pred == 1))
    false_negatives = torch.logical_and((y_true == 1), (y_pred == 0))
    return true_positives / (true_positives + false_negatives)

def f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """F1: 2 * (Precision * Recall) / (Precision + Recall)"""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r)

def pitch_number_acc(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """The accuracy at which y_pred has the same number of 1s as y_true."""
    y_pred = torch.round(y_pred)
    num_true_ones = y_true.sum(1)
    num_pred_ones = y_pred.sum(1)
    return (num_true_ones == num_pred_ones).sum() / len(num_true_ones)

class NStringChordAccuracy(Metric):
    """ Computes accuracy across an epoch on chords with n strings. Ignores 
    NaNs.
    
    `n`: number of chords to consider. also considers 'multi', which includes 
        all chords with n>1 strings 
    """
    def __init__(self, n):
        assert (type(n) == int) or (n == 'multi')
        self.n = n
    
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        strings_per_y = torch.sum(y_true, axis=1)
        
        if self.n == 'multi':
            pitches = (strings_per_y > 1)
        else:
            pitches = (strings_per_y == self.n)
        
        y_true = y_true[pitches]
        y_pred = y_pred[pitches]
        
        return categorical_accuracy(y_true, y_pred)