from typing import Tuple, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from ppo import transformer as trf


class EntitiesEncoder(keras.layers.Layer):
    def __init__(self, embed_dim=2 ** 7, num_heads=8, ff_dim=128, rate=0.0, name=None, **args):
        super(EntitiesEncoder, self).__init__(name=name, *args)
        self.supports_masking = True
        self.embed_dim = embed_dim
        self.dense = keras.layers.Dense(embed_dim)
        self.transformer_block = trf.TransformerBlock(embed_dim, num_heads, ff_dim, rate)
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None, mask=None):
        if mask is not None and mask.dtype != inputs.dtype:
            mask = tf.cast(mask, inputs.dtype)
        inputs = self.layernorm(inputs)
        inputs = self.dense(inputs)
        return self.transformer_block(inputs, training, mask=mask)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], self.embed_dim

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.embed_dim

    def get_config(self):
        config = super(EntitiesEncoder, self).get_config()
        config.update({
            "dense": self.dense,
            "transformer_block": self.transformer_block,
            "layernorm": self.layernorm,
            'embed_dim': self.embed_dim
        })
        return config


def build_shared(units, scalars) -> keras.layers.LSTM:
    scalars = keras.layers.LayerNormalization()(scalars)

    # units_encoder
    units_encoder = EntitiesEncoder(embed_dim=2 ** 7, num_heads=4, ff_dim=128, name='entities_encoder')(units)

    # scalars encoder
    scalars_encoder = keras.layers.LayerNormalization()(scalars)
    scalars_encoder = keras.layers.Dense(64, activation='relu', name='scalars_encoder1')(scalars_encoder)
    scalars_encoder = keras.layers.Dense(64, activation='relu', name='scalars_encoder2')(scalars_encoder)
    # combine scalars and units
    scalars_encoder = keras.layers.RepeatVector(units_encoder.shape[1])(scalars_encoder)
    encoder = keras.layers.concatenate([units_encoder, scalars_encoder], axis=-1)
    lstm_out = keras.layers.LSTM(128, name='DeepLSTM')(encoder)
    return lstm_out


def build_actor(verbose=True, lr=1e-4):
    n_actions = 19
    # create the model architecture

    # inputs
    units_input = keras.layers.Input(shape=(22, 8), name='units_input')
    scalars_input = keras.layers.Input(shape=(31,), name='scalars_input')

    # advantage and old_prediction inputs
    advantage = keras.layers.Input(shape=(1,), name='advantage')
    old_action = keras.layers.Input(shape=(n_actions,), name='old_action')

    action_lbl = keras.layers.Input(shape=(n_actions,), name='action_lbl')
    # action_pred = keras.layers.Input(shape=(n_ships, n_actions), name='action_pred')

    # build_shared
    encoder = build_shared(units_input, scalars_input)

    # outputs
    action = keras.layers.Dense(n_actions, activation=keras.activations.softmax)(encoder)
    inputs = [units_input, scalars_input, advantage, old_action, action_lbl]

    model = keras.models.Model(inputs, action)
    model.add_loss(ppo_loss(action_lbl, action, advantage, old_action))

    model.compile(optimizer=keras.optimizers.Adam(lr))
    if verbose: model.summary()
    return model


def build_critic(verbose=True, lr=1e-4):
    # inputs
    units_input = keras.layers.Input(shape=(22, 8), name='units_input')
    scalars_input = keras.layers.Input(shape=(31,), name='scalars_input')

    # build_shared
    encoder = build_shared(units_input, scalars_input)

    # outputs
    value_dense = keras.layers.Dense(1, name='value')(encoder)
    inputs = [units_input, scalars_input]

    model = keras.models.Model(inputs, value_dense)

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr))
    if verbose: model.summary()
    return model


LOSS_CLIPPING = 2  # Only implemented clipping for the surrogate loss, paper said it was best
ENTROPY_LOSS = 5e-3
GAMMA = 0.99


def ppo_loss(label_layer, prediction_layer, advantage, old_prediction, clip=False):
    # label_layer = tf.reshape(label_layer, (K.shape(label_layer)[0], -1))
    # prediction_layer = tf.reshape(prediction_layer, (K.shape(prediction_layer)[0], -1))
    # old_prediction = tf.reshape(old_prediction, (K.shape(old_prediction)[0], -1))

    prob = label_layer * prediction_layer
    old_prob = label_layer * old_prediction
    r = prob / (old_prob + 1e-10)
    clipped = r
    if clip:
        clipped = K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING)
    return -K.mean(K.minimum(r * advantage,
                             clipped
                             * advantage) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10)))

# build_actor()
# build_critic()
