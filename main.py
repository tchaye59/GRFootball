# Set up the Environment.
from gfootball.env.wrappers import *
from kaggle_environments import make

env = make("football",
           configuration={"save_video": False, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True})

obs = env.reset()
done = True

obs = obs[0].observation.players_raw[0]

l_units = [[x[0] for x in obs['left_team']], [x[1] for x in obs['left_team']],
           [x[0] for x in obs['left_team_direction']], [x[1] for x in obs['left_team_direction']],
           obs['left_team_tired_factor'], obs['left_team_yellow_card'],
           obs['left_team_active'], obs['left_team_roles']]

l_units = np.r_[l_units].T.astype(np.float32)

r_units = [[x[0] for x in obs['right_team']], [x[1] for x in obs['right_team']],
           [x[0] for x in obs['right_team_direction']], [x[1] for x in obs['right_team_direction']],
           obs['right_team_tired_factor'],
           obs['right_team_yellow_card'],
           obs['right_team_active'], obs['right_team_roles']]

r_units = np.r_[r_units].T.astype(np.float32)

game_mode = [0 for _ in range(6)]
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


while not done:
    state, reward, done, info = env.step(1)
    print('reward ', reward)
