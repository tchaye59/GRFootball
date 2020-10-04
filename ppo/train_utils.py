import os
import threading
import uuid
from queue import Queue
import dill
import numpy as np
from scipy.signal import lfilter
import tensorflow as tf
from foot_util import FootEnv
from ppo.model_utils import GAMMA
from ppo.policy import Policy
from threading import Lock

lock = Lock()


class Memory:
    def __init__(self):
        # inputs
        self.units = []
        self.scalars = []
        # action
        self.actions_matrix = []
        self.actions_probs = []
        # rewards
        self.rewards = []
        self.terminal = []

    def isEmpty(self):
        return len(self.rewards) == 0

    def store(self, obs, actions, reward, done):
        # inputs
        units, saclars = obs
        self.units.append(units)
        self.scalars.append(saclars)

        # actions
        _, actions_matrix, actions_probs = actions
        if actions_matrix is not None: self.actions_matrix.append(actions_matrix)
        if actions_probs is not None: self.actions_probs.append(actions_probs)
        # reward
        self.rewards.append(reward)
        self.terminal.append(done)

    def discount(self, x, gamma=GAMMA):
        return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def discount_rewards(self, GAMMA=0.99):
        return self.discount(self.rewards, GAMMA)

    def normalize(self, x):
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / np.maximum(std, 1e-6)

    def compute_advantages(self, pred_value, GAMMA=0.99, LAMBDA=0.95, normalize=True):
        # Computes GAE (generalized advantage estimations (from the Schulman paper))
        rewards = np.array(self.rewards, dtype=np.float32)
        pred_value_t = pred_value
        pred_value_t1 = np.concatenate([pred_value[1:], [0.]])
        pred_value_t1[self.terminal] = 0
        advantage = rewards + GAMMA * pred_value_t1 - pred_value_t
        advantage = self.normalize(self.discount(advantage, GAMMA * LAMBDA))
        return np.array(self.discount_rewards(), dtype=np.float32), \
               advantage.astype(np.float32)

    def compute_normal_advantages(self, pred_value, GAMMA=0.99):
        rewards = np.array(self.discount_rewards(GAMMA), dtype=np.float32)
        advantage = rewards - pred_value
        return rewards.astype(np.float32), advantage.astype(np.float32)

    def get_all_as_tensors(self):
        rewards = np.array(self.discount_rewards(), dtype=np.float32)
        units = tf.concat(self.units, axis=0)
        scalars = tf.concat(self.scalars, axis=0)

        actions_matrix = tf.convert_to_tensor(self.actions_matrix, dtype=tf.float32)
        actions_probs = tf.convert_to_tensor(self.actions_probs, dtype=tf.float32)
        dones = np.array(self.terminal, dtype=np.float32)
        return (units, scalars), actions_matrix, actions_probs, rewards, dones


class EpisodeCollector(threading.Thread):
    n_episode = 0
    reward_sum = 0
    max_episode = 0

    def __init__(self, env: FootEnv, policy: Policy, result_queue=None, replays_dir=None):
        super().__init__()
        self.result_queue = result_queue
        self.env = env
        self.policy = policy
        self.replays_dir = replays_dir
        self.n_episode = -1

    def clone(self):
        obj = EpisodeCollector(self.env, self.policy)
        obj.result_queue = self.result_queue
        obj.replays_dir = self.replays_dir
        obj.n_episode = self.n_episode
        return obj

    def run(self):
        self.result_queue.put(self.collect(1))

    def collect(self, n=1):
        n = max(n, self.n_episode)
        return [self.collect_()[0] for _ in range(n)]

    def collect_(self):
        memory = Memory()
        done = False
        EpisodeCollector.n_episode += 1
        obs = self.env.reset()
        i = 0
        total_reward = 0
        state = None
        while not done:
            actions, state = self.policy.get_action(obs, state=state)
            new_obs, reward, done, info = self.env.step(actions[0])
            total_reward = reward
            # store data
            memory.store(obs, actions, reward, done)

            if done or i % 100 == 0:
                with lock:
                    print(
                        f"Episode: {EpisodeCollector.n_episode}/{EpisodeCollector.max_episode} | "
                        f"Step: {i} | "
                        f"Env ID: {self.env.env_id} | "
                        f"Reward: {total_reward} | "
                        f"Done: {done} | "
                        f"Total Rewards: {EpisodeCollector.reward_sum} | "
                    )
                    print(info)

            obs = new_obs
            i += 1
        EpisodeCollector.reward_sum += total_reward
        if self.replays_dir:
            with open(os.path.join(self.replays_dir, f'replay-{uuid.uuid4().hex}.dill'), 'wb') as f:
                dill.dump(memory, f)
        return [memory]


class ParallelEpisodeCollector:

    def __init__(self, env_fn, n_jobs, policy: Policy, replays_dir=None, ):
        self.n_jobs = n_jobs
        self.policy: Policy
        self.envs = []
        self.result_queue = Queue()
        self.replays_dir = replays_dir
        for i in range(n_jobs):
            self.envs.append(env_fn(env_id=i))
        self.collectors = [EpisodeCollector(env,
                                            policy=policy,
                                            result_queue=self.result_queue,
                                            replays_dir=replays_dir) for env in self.envs]

    def collect(self, n_steps=1):
        if not n_steps: n_steps = 1
        result_queue = self.result_queue
        for i, collector in enumerate(self.collectors):
            collector = collector.clone()
            self.collectors[i] = collector
            collector.n_episode = max(1, int(n_steps / len(self.collectors)))
            print("Starting collector {}".format(i))
            collector.start()
        tmp = []
        for _ in self.collectors:
            res = result_queue.get()
            tmp.extend(res)
        [collector.join() for collector in self.collectors]
        return tmp


def env_fn(env_id=1):
    return FootEnv(env_id=env_id)


if __name__ == '__main__':
    policy = Policy()
    # collector = ParallelEpisodeCollector(env_fn, 2, policy)
    collector = EpisodeCollector(env_fn(), policy)
    memories = collector.collect(1)
    policy.train(memories)
