from __future__ import division

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

#from catch_ball import CatchBall
from invador import Invador
from dqn_agent import DQNAgent

if __name__ == "__main__":
    # environmet, agent
    env = Invador(simple=False, plot=True)
    obs = env.reset()
    agent = DQNAgent(env.enable_actions, [obs.shape[0], obs.shape[1]], env.name)
    agent.load_model()

    while True:
        state, reward, terminal = env.observe()
        env.update_plot()

        if terminal:
            REWARD = reward
            print("REWARD: %.03d"%REWARD)
            env.reset()
        else:
            action = agent.select_action(state.reshape(env.IMG_SIZE*env.IMG_SIZE), 0.0)
            env.step(action)
