import gym
import numpy as np

from stable_baselines import TD3
from stable_baselines.ddpg import NormalActionNoise

def test_td3():
    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # train_freq=1, gradient_steps=1
    model = TD3('MlpPolicy', 'Pendulum-v0', action_noise=action_noise, learning_starts=1000, verbose=1)
    model.learn(100000)
