import gym
import numpy as np
from kaggle_environments import make

# from agents import rule1

# from kaggle_environments.envs.football.helpers import *
from agents import rule1


class FootEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, right_agent='/home/shell/PycharmProjects/GRFootball/agents/rule1.py', env_id=0):
        super(FootEnv, self).__init__()
        # right_agent = '/home/shell/PycharmProjects/GRFootball/agents/random.py'
        self.env_id = env_id
        self.agents = [None, right_agent]
        self.env = make("football", configuration={"save_video": False,
                                                   "scenario_name": "11_vs_11_kaggle",
                                                   "running_in_notebook": False})
        self.trainer = None
        self.config = self.env.configuration
        self.obs = None

        # self.action_shape = self.env.action_space
        shape = []
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    def step(self, action):
        # act
        obs, reward, done, info = self.trainer.step([action])
        obs = obs['players_raw'][0]
        self.obs = obs
        return OBSParser.parse(obs), reward, done, info

    def reset(self):
        self.trainer = self.env.train(self.agents)
        obs = self.trainer.reset()
        obs = obs['players_raw'][0]
        self.obs = obs
        return OBSParser.parse(obs)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        pass


class OBSParser(object):

    @staticmethod
    def parse(obs):
        l_units = [[x[0] for x in obs['left_team']], [x[1] for x in obs['left_team']],
                   [x[0] for x in obs['left_team_direction']], [x[1] for x in obs['left_team_direction']],
                   obs['left_team_tired_factor'], obs['left_team_yellow_card'],
                   obs['left_team_active'], obs['left_team_roles']]

        l_units = np.r_[l_units].T

        r_units = [[x[0] for x in obs['right_team']], [x[1] for x in obs['right_team']],
                   [x[0] for x in obs['right_team_direction']], [x[1] for x in obs['right_team_direction']],
                   obs['right_team_tired_factor'],
                   obs['right_team_yellow_card'],
                   obs['right_team_active'], obs['right_team_roles']]

        r_units = np.r_[r_units].T

        units = np.r_[l_units, r_units].astype(np.float32)

        game_mode = [0 for _ in range(7)]
        game_mode[obs['game_mode']] = 1
        scalars = [*obs['ball'],
                   *obs['ball_direction'],
                   *obs['ball_rotation'],
                   obs['ball_owned_team'],
                   obs['ball_owned_player'],
                   *obs['score'],
                   obs['steps_left'],
                   *game_mode,
                   *obs['sticky_actions']]

        scalars = np.r_[scalars].astype(np.float32)
        return units[np.newaxis, :], scalars[np.newaxis, :]


if __name__ == "__main__":
    env = FootEnv()
    state = env.reset()
    done = False
    while not done:
        state, reward, done, info = env.step(5)
        print('reward ', reward, info)
