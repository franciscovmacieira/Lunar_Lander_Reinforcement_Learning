import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np

class LunarLanderModified(LunarLander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        reward -= abs(pos_x)

        # Penalize for high velocities
        reward -= (abs(vel_x) + abs(vel_y))

        # Penalize for large angles and angular velocity
        reward -= (abs(angle) + abs(angular_vel))

        # Reward for leg contact (partial landing)
        if left_leg_contact or right_leg_contact:
            reward += 10.0

        # Additional reward for successful landing
        if terminated and (left_leg_contact and right_leg_contact):
            reward += 100.0

        return state, reward, terminated, truncated, info

