import os
import random
import time
import numpy as np
import tensorflow as tf
from ppo.model_utils import build_actor, build_critic

N_ACTIONS = 19
LR = 0.0001
BATCH_SIZE = 1024
EPOCHS = 10
GAMMA = 0.99
LAMBDA = 0.95


def random_weighted_choice(weights):
    weights += 0.1
    res = random.choices(np.arange(weights.size), weights=weights, k=1)
    return res[0]


class Policy:
    def __init__(self, val=False):
        self.actor = build_actor(LR)
        self.critic = build_critic(LR)

        self.val = val

    def get_values(self, X):
        return self.critic.predict(X).flatten()

    def get_action(self, X, state):
        action_prob = self.actor.predict(X)
        action_prob = action_prob[0]
        # action_probs = np.nan_to_num(action_probs[0])
        n_actions = action_prob.size
        if self.val:
            action = np.argmax(action_prob, axis=-1)
        else:
            action = np.random.choice(n_actions, p=action_prob)

        # matrix
        action_matrix = np.zeros(n_actions, np.float32)
        action_matrix[action] = 1

        return (action, action_matrix, action_prob), state

    def train(self, memories):
        actor_ds, critic_ds = None, None
        # prepare dataset
        for i, memory in enumerate(memories):
            print(f"Add Memory {i + 1}/{len(memories)}")
            inputs, actions_matrix, actions_probs, rewards, dones = memory.get_all_as_tensors()
            c_inputs = inputs
            pred_values = self.get_values(c_inputs)

            # Generalized Advantage Estimation
            rewards, advantage = memory.compute_advantages(pred_values)
            rewards = rewards[:, np.newaxis]
            advantage = advantage[:, np.newaxis]

            labels = actions_matrix
            a_inputs = *inputs, advantage, actions_probs, labels

            if actor_ds is None:
                actor_ds = tf.data.Dataset.from_tensor_slices((a_inputs, labels))
            else:
                actor_ds = actor_ds.concatenate(tf.data.Dataset.from_tensor_slices((a_inputs, labels)))
            if critic_ds is None:
                critic_ds = tf.data.Dataset.from_tensor_slices((c_inputs, rewards))
            else:
                critic_ds = critic_ds.concatenate(tf.data.Dataset.from_tensor_slices((c_inputs, rewards)))

        # train
        print("Updating")
        actor_ds = actor_ds.shuffle(100).batch(BATCH_SIZE).prefetch(2)
        critic_ds = critic_ds.shuffle(100).batch(BATCH_SIZE).prefetch(2)

        s = time.time()
        a_losses = self.actor.fit(actor_ds, epochs=EPOCHS, verbose=False)
        a_time = time.time() - s
        print(f">>>{a_time}")
        s = time.time()
        c_losses = self.critic.fit(critic_ds, epochs=EPOCHS, verbose=False)
        c_time = time.time() - s
        print(f">>>{c_time}")
        print(f"Duration: {a_time + c_time}")

        return a_losses.history['loss'], c_losses.history['loss']

    def save(self, path):
        self.actor.save_weights(path + '.actor.h5')
        self.critic.save_weights(path + '.critic.h5')

    def load(self, path):
        if os.path.exists(path + '.actor.h5') or os.path.exists(path + '.critic.h5'):
            self.actor.load_weights(path + '.actor.h5')
            self.critic.load_weights(path + '.critic.h5')
