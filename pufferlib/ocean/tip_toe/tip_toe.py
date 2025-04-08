import numpy as np
from gymnasium import spaces

import pufferlib
from pufferlib.ocean.tip_toe.cy_tip_toe import CyTipToe


class TipToeEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1, buf=None, 
                 grid_size_x=10, grid_size_y=10, num_agents=10, max_steps=600):
        # Env Setup
        self.render_mode = render_mode
        self.report_interval = report_interval

        # Validate num_agents
        if not isinstance(num_agents, int) or num_agents <= 0:
            raise ValueError("num_agents must be an integer greater than 0.")
        self.num_agents = num_envs * num_agents
        self.num_agents_per_env = num_agents

        if not isinstance(max_steps, int) or max_steps < 10:
            raise ValueError("max_steps must be an int >= 10")
        self.max_steps = max_steps

        min_grid_size = 5

        if not isinstance(grid_size_x, int) or grid_size_x < min_grid_size:
            raise ValueError(
                f"grid_size_x must be an integer >= {min_grid_size}. "
                f"Received grid_size_x = {grid_size_x}"
            )
        self.grid_size_x = grid_size_x

        if not isinstance(grid_size_y, int) or grid_size_y < min_grid_size:
            raise ValueError(
                f"grid_size_y must be an integer >= {min_grid_size}. "
                f"Received grid_size_y = {grid_size_y}"
            )
        self.grid_size_y = grid_size_y
        
        # 2D Local crop obs space
        self.num_obs = (grid_size_x * grid_size_y) + (2 * num_agents);

        self.single_observation_space = spaces.Box(low=0, high=1,
            shape=(self.num_obs,), dtype=np.int8)
        self.single_action_space = spaces.Discrete(9)

        super().__init__(buf=buf)
        self.c_envs = CyTipToe(self.observations, self.actions, self.rewards, self.terminals, num_envs, num_agents, 
                                    grid_size_x, grid_size_y, max_steps)

    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()
        
    def close(self):
        self.c_envs.close() 

def test_performance(timeout=10, atn_cache=1024):
    env = TipToeEnv(num_envs=1024, grid_size_x=10, grid_size_y=10, num_agents=10, max_steps=600)
 
    env.reset()
    tick = 0

    actions = np.random.randint(0, 9, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: {env.num_agents * tick / (time.time() - start)}')

if __name__ == '__main__':
    test_performance()
