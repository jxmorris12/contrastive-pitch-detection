import keras
import tensorflow as tf

# precision = tf.keras.metrics.Precision()
# recall = tf.keras.metrics.Recall()
# def f1_score(y_true, y_pred):
#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     return 2 * ((p * r) / (p + r))
    
    
def _euclidean_distance(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return torch.sqrt(torch.sum((y_pred - y_true)**2, axis=-1))

def _acc(y_true, y_pred):
    """ computes the accuracy of predictions ``y_pred`` given labels ``y_true``
    
    this is how the keras backend does it. see:
    datascience.stackexchange.com/questions/14415
    """
    label_slots = (y_true == 1)
    return torch.mean(torch.equal(y_true[label_slots], torch.round(y_pred)[label_slots]))

def pitch_number_acc(y_true, y_pred):
    """ whether the number of activated pitches is the correct number """
    y_pred = torch.round(y_pred)
    num_true_pitches = torch.sum(y_true, axis=1)
    num_pred_pitches = torch.sum(y_pred, axis=1)
    return torch.mean(torch.equal(num_true_pitches, num_pred_pitches))
    

class NStringChordAccuracy(keras.metrics.Metric):
    """ Computes accuracy across an epoch on chords with n strings. Ignores 
    NaNs.
    
    `n`: number of chords to consider. also considers 'multi', which includes 
        all chords with n>1 strings 
    """
    def __init__(self, n, **kwargs):
        assert (type(n) == int) or (n == 'multi')
        super().__init__(name=f'chord_{n}_string_accuracy', **kwargs)
        self.n = n
        self.total_acc = 0.0
        self.total_samples = 0.0
    
    def update_state(self, y_true, y_pred):
        strings_per_y = torch.sum(y_true, axis=1)
        
        if self.n == 'multi':
            pitches = (strings_per_y > 1)
        else:
            pitches = (strings_per_y == self.n)
        
        y_true = y_true[pitches]
        y_pred = y_pred[pitches]
        
        self.total_acc += _acc(y_true, y_pred)
        self.total_samples += 1
    
    def result(self):
        return torch.divide(self.total_acc, self.total_samples)

    def reset(self):
        self.total_acc = 0.0
        self.total_samples = 0.0