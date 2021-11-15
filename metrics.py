import keras
import tensorflow as tf

from keras import backend as K

precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r))

def _reshape_lstm_outputs(y_true, y_pred):
    """ This flattens tensors that include a timesteps dimension so that they
    no longer include that dimension. """
    if len(K.shape(y_true)) == 3:
        D = K.shape(y_true)[-1]
        y_true, y_pred = K.reshape(y_true, [-1, D]), K.reshape(y_pred, [-1, D])
    return y_true, y_pred

def MeanSquaredError(y_true, y_pred):
    return tf.reduce_mean((y_pred - y_true)**2)
    
def loss_1(y_true, y_pred):
    """ This loss function uses binary cross-entropy to solve pitch detection
    as a multiclass classification problem. It basically expects a classifier
    for each possible MIDI value and treats each one as an independent binary
    classifier.
    """
    # breakpoint()
    _reshape_lstm_outputs(y_true, y_pred)
    
    return K.binary_crossentropy(y_true, y_pred)


def string_level_regression_loss(y_true, y_pred):
    """ The earliest versions of CLAPTON were trained to predict a binary 
    yes-or-no answer. The model output was a vector of probabilities starting
    with the probability associated with the minimum MIDI value 'listened for' 
    and ending with the probability associated with the maximum MIDI value.
    
    This loss function takes a different approach. The model must output six
    values, each predicting the MIDI value a maximum of six concurrent notes 
    played.
    
    This loss function takes the original model ``y_true``, a vector of
    an array of N (1<=N<=6) MIDI values outputted. Then, normal regression 
    is used on the first N outputs of the network.
    """
    nonzero_mask = K.greater(y_true, K.constant(0))
    nonzero_idxs = tf.where(K.flatten(nonzero_mask))
    y_true = K.gather(K.flatten(y_true), nonzero_idxs)
    y_pred = K.gather(K.flatten(y_pred), nonzero_idxs)
    return K.mean(_euclidean_distance(y_true, y_pred))
    
    
def _euclidean_distance(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def _acc(y_true, y_pred):
    """ computes the accuracy of predictions ``y_pred`` given labels ``y_true``
    
    this is how the keras backend does it. see:
    datascience.stackexchange.com/questions/14415
    """
    label_slots = (y_true == 1)
    return K.mean(K.equal(y_true[label_slots], K.round(y_pred)[label_slots]))

def pitch_number_acc(y_true, y_pred, mse=False):
    """ whether the number of activated pitches is the correct number """
    if len(K.shape(y_true)) == 3:
        D = K.shape(y_true)[-1]
        y_true, y_pred = K.reshape(y_true, [-1, D]), K.reshape(y_pred, [-1, D])
    y_pred = K.round(y_pred)
    num_true_pitches = K.sum(y_true, axis=1)
    num_pred_pitches = K.sum(y_pred, axis=1)
    if mse:
        return _euclidean_distance(num_true_pitches, num_pred_pitches)
    else:
        return K.mean(K.equal(num_true_pitches, num_pred_pitches))
    

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
        self.total_acc = self.add_weight(name='total_samples', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        _reshape_lstm_outputs(y_true, y_pred)
        strings_per_y = K.sum(y_true, axis=1)
        
        if self.n == 'multi':
            pitches = (strings_per_y > 1)
        else:
            pitches = (strings_per_y == self.n)
        
        y_true = y_true[pitches]
        y_pred = y_pred[pitches]
        
        acc = _acc(y_true, y_pred)
        
        if not tf.math.is_nan(acc):
            self.total_acc.assign_add(acc)
            self.total_samples.assign_add(1)
    
    def result(self):
        return tf.divide(self.total_acc, self.total_samples)

    def reset_states(self):
        self.total_acc.assign(0)
        self.total_samples.assign(0)


def f1(y_true, y_pred):
    """ F1 = 2 (P * R) / (P + R) """
    precision = tf.keras.metrics.Precision()(y_true, y_pred)
    recall = tf.keras.metrics.Recall()(y_true, y_pred)
    num = tf.math.multiply(K.constant(2, dtype=tf.float32), tf.math.multiply(precision, recall))
    den = tf.math.add(precision, recall)
    return tf.math.divide(num, den)