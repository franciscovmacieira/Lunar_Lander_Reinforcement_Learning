import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np

class LunarLanderModified(LunarLander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _reset_env(self, seed=None):
        # Chama o método original para criar o mundo, o lander, etc.
        super()._reset_env(seed=seed)

        # Defina uma nova largura do helipad menor que o original para aproximar as bandeiras.
        # O valor original de PAD_WIDTH no LunarLander é algo em torno de 14.0.
        # Vamos reduzir, por exemplo, para 8.0.
        new_pad_width = 11.0
        self.helipad_x1 = self.helipad_x - new_pad_width / 2
        self.helipad_x2 = self.helipad_x + new_pad_width / 2

        # Remover as bandeiras criadas pela classe pai, pois elas foram colocadas com o PAD_WIDTH original.
        for flag in self.flags:
            self.world.DestroyBody(flag)
        self.flags = []

        # Recriar as bandeiras nas novas posições mais próximas
        for x in [self.helipad_x1, self.helipad_x2]:
            flag = self.world.CreateStaticBody(
                fixtures=fixtureDef(
                    shape=polygonShape(box=(0.5, 1.0)),
                    density=0.0,
                    friction=0.1,
                ),
                position=(x, self.helipad_y)
            )
            self.flags.append(flag)

    def step(self, action):
        # Perform the action in the environment
        state, reward, terminated, truncated, info = super().step(action)

        # Extract state variables
        pos_x, pos_y = state[0], state[1]        # Position
        vel_x, vel_y = state[2], state[3]        # Velocity
        angle, angular_vel = state[4], state[5]  # Angle and angular velocity
        left_leg_contact, right_leg_contact = state[6], state[7]  # Leg contact

        # Modify the reward function

        # Penalize for moving away from the center
        reward -= (abs(pos_x))/4

        # Penalize for high velocities
        reward -= ((abs(vel_x) + abs(vel_y)))/4

        # Penalize for large angles and angular velocity
        reward -= ((abs(angle) + abs(angular_vel)))/4

        # Reward for leg contact (partial landing)
        if left_leg_contact or right_leg_contact:
            reward += 30.0

        # Additional reward for successful landing
        if terminated and (left_leg_contact and right_leg_contact):
            reward += 300.0

        return state, reward, terminated, truncated, info

