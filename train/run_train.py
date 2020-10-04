import json
import os
from typing import List
import tensorflow as tf
from ppo.policy import Policy
from ppo.train_utils import ParallelEpisodeCollector, EpisodeCollector, env_fn, Memory
import multiprocessing as mp

PARALLEL_COLLECTOR = False

NAME = 'foot'
path = os.path.join('data', NAME)
tf_logs_path = os.path.join(path, 'tf_log')
info_path = os.path.join(path, 'info.json')
writer = tf.summary.create_file_writer(tf_logs_path)
os.makedirs(tf_logs_path, exist_ok=True)
current_step = 0
# create policy
policy_path = os.path.join(path, 'model')
ma_policy_path = os.path.join(path, 'model_ma')
val_policy_path = os.path.join(path, 'model_val')
policy = Policy()
policy.load(policy_path)
collector = None
global_ma_reward = 0.
best_ma_reward = 0.
best_val_reward = 0.
n_episodes = 0
n_collect = 5

# restore training metadata
if os.path.exists(info_path):
    with open(info_path, 'r') as f:
        info = json.load(f)
        current_step = info['current_step']
        global_ma_reward = info['global_ma_reward']
        best_ma_reward = info['best_ma_reward']
        n_episodes = info['n_episodes']
        best_val_reward = info['best_val_reward']
        assert NAME == info['name']

# Define the experience collector
if PARALLEL_COLLECTOR:
    collector = ParallelEpisodeCollector(env_fn, mp.cpu_count(), policy)
else:
    collector = EpisodeCollector(env_fn(), policy)


def train(steps):
    global current_step, global_ma_reward, best_ma_reward, best_val_reward

    max_step = steps
    EpisodeCollector.max_episode = max_step
    EpisodeCollector.n_episode = n_episodes
    i = 0
    while current_step < max_step:
        memories = collector.collect(n_collect)
        print("Updating the policy...")
        losses = policy.train(memories)
        policy.save(policy_path)
        record(memories, current_step, losses)
        if global_ma_reward >= best_ma_reward:
            best_ma_reward = global_ma_reward
            print("Saving best policy...")
            policy.save(ma_policy_path)
        print(
            f"Episode: {n_episodes} | "
            f"Moving Average Reward: {int(global_ma_reward)} | "
            f"Best Moving Average Reward: {int(best_ma_reward)} | "
            f"Episode Rewards: {[sum(mem.rewards) for mem in memories]} | "
        )

        # Validation
        if i % 10 == 0 and i != 0:
            val_collector = collector
            print("Agent validation...")
            policy.val = True
            memories = val_collector.collect(10)
            policy.val = False

            EpisodeCollector.n_episode -= len(memories)
            rew = sum([sum(mem.rewards) for mem in memories if not mem.isEmpty()]) / len(memories)
            print(f"Validation reward : {rew}")
            with writer.as_default():
                tf.summary.scalar("val_reward", rew, step=current_step)
                writer.flush()
            if rew >= best_val_reward:
                best_val_reward = rew
                print("Saving best validation policy...")
                policy.save(val_policy_path)

        current_step += 1
        i += 1


def record(memories: List[Memory], current_step, losses):
    global global_ma_reward, n_episodes, info_path
    n_episodes += len(memories)
    global_ma_reward = sum([sum(memory.rewards) for memory in memories]) / len(memories)

    with writer.as_default():
        tf.summary.scalar("global_ma_reward", global_ma_reward, step=current_step)
        if losses[0] is not None: tf.summary.scalar("Actor loss", sum(losses[0]) / len(losses[0]), step=current_step)
        if losses[1] is not None: tf.summary.scalar("Critic loss", sum(losses[1]) / len(losses[1]), step=current_step)
        tf.summary.scalar("best_ma_reward", best_ma_reward, step=current_step)
        # tf.summary.scalar("best_val_reward", best_val_reward, step=current_step)
        writer.flush()

    with open(info_path, 'w') as f:
        json.dump({
            'current_step': current_step,
            'global_ma_reward': global_ma_reward,
            'best_ma_reward': best_ma_reward,
            'n_episodes': n_episodes,
            'best_val_reward': best_val_reward,
            'name': NAME,
        }, f)


def exponential_average(old, new, b1):
    return old * b1 + (1 - b1) * new


# Start training
train(100000)
