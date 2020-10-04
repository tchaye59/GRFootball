import multiprocessing as mp
import os
from tools import constantes as cst
from tools.ppo.ac_policy import ACPolicy
from tools.train.train_utils import ParallelEpisodeCollector, EpisodeCollector, WarEnv

PARALLEL_COLLECTOR = False
NAME = 'policyRed_vs_ruleBlue'
path = os.path.join(cst.DATA_PATH, NAME)
# create policy
red_policy_path = os.path.join(path, 'red_policy_model')
red_ma_policy_path = os.path.join(path, 'red_ma_policy_model')
red_val_policy_path = os.path.join(path, 'red_val_policy_model')
red_policy = ACPolicy('red', val=True)
red_policy.load(red_policy_path)


# Define the experience collector
def init_collectors():
    if PARALLEL_COLLECTOR:
        n = mp.cpu_count() - 1
        n = min(n, 10)
        print(f"Running on {n} envs")
        return ParallelEpisodeCollector([cst.config], n, red_policy, None)
    else:
        return EpisodeCollector(WarEnv(**cst.config, ), red_policy, None)


collector = init_collectors()

red_memories, _ = collector.collect()
for memory in red_memories:
    print(memory.rewards[-1])
