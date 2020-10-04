import random

from kaggle_environments.envs.football.helpers import human_readable_agent


@human_readable_agent
def agent(obs):
    return random.choice(range(19))
