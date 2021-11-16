import os
import tensorflow as tf

from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, \
    LayerNormalization, MaxPool2D, Dropout, Permute, Flatten, Dense, \
    TimeDistributed, LSTM, concatenate
from tensorflow.keras.models import Model

# store as a global variable, since we only support a few models for now
models = {
    'tiny': None,
    'small': None,
    'medium': None,
    'large': None,
    'full': None
}

# the model is trained on 16kHz audio
model_srate = 16000

CREPE_MODEL_CAPACITIES = {
    'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
}

def CREPE(model_capacity, input_dim=1024, num_output_nodes=1, load_pretrained=False,
  freeze_some_layers=False, add_intermediate_dense_layer=False,
    add_dense_output=True, out_activation='sigmoid'):
    """
    Build the CNN model and load the weights
    Parameters
    ----------
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity, which determines the model's
        capacity multiplier to 4 (tiny), 8 (small), 16 (medium), 24 (large),
        or 32 (full). 'full' uses the model size specified in the paper,
        and the others use a reduced number of filters in each convolutional
        layer, resulting in a smaller model that is faster to evaluate at the
        cost of slightly reduced pitch estimation accuracy.
        
        
        Model paths in /p:
        - /p/qdata/jm8wx/other/crepe/crepe/model-full.h5
        - /p/qdata/jm8wx/other/crepe/crepe/model-large.h5
        - /p/qdata/jm8wx/other/crepe/crepe/model-medium.h5
        - /p/qdata/jm8wx/other/crepe/crepe/model-small.h5
        - /p/qdata/jm8wx/other/crepe/crepe/model-tiny.h5
    
    
    TODO this should probably be a class, not a function, at this point.

    Returns
    -------
    model : tensorflow.keras.models.Model
        The pre-trained keras model loaded in memory
    """
    
    assert model_capacity in CREPE_MODEL_CAPACITIES, f'unknown model capacity {model_capacity}'
    
    if load_pretrained:
        # assert input_dim == 1024, "pretrained models require input dimension 1024"
        if input_dim != 1024:
          print('WARNING: got load_pretrained=True but input_dim!=1024. not loading pretrained model')
          load_pretrained = False

    capacity_multiplier = CREPE_MODEL_CAPACITIES[model_capacity]

    layers = [1, 2, 3, 4, 5, 6]
    filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(input_dim,), name='input', dtype='float32')
    y = Reshape(target_shape=(input_dim, 1, 1), name='input-reshape')(x)

    for l, f, w, s in zip(layers, filters, widths, strides):
        y = Conv2D(f, (w, 1), strides=s, padding='same',
                   activation='relu', name=f"conv{l}")(y)
        y = BatchNormalization(name=f"conv{l}-BN")(y)
        y = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                      name=f"conv{l}-maxpool")(y)
        y = Dropout(0.25, name=f"conv{l}-dropout")(y)

    y = Permute((2, 1, 3), name="transpose")(y)
    y = Flatten(name="flatten")(y)
    
    if load_pretrained or add_intermediate_dense_layer:
        y = Dense(360, activation='relu', name="classifier")(y) # 'cents' from CREPE
    
    if add_dense_output:
        y = Dense(num_output_nodes, activation=out_activation, name="out")(y)
    
    model = Model(inputs=x, outputs=y)

    if load_pretrained:
        package_dir = '/p/qdata/jm8wx/other/crepe/crepe'
        filename = 'model-{}.h5'.format(model_capacity)
        model.load_weights(os.path.join(package_dir, filename), by_name=True)

        if freeze_some_layers:
            layers_to_freeze = model.layers[:0]
            print(f'Freezing {len(layers_to_freeze)} layers.')
            for layer in layers_to_freeze: layer.trainable = False
        else:
            print('not freezing any layers!')

    return model