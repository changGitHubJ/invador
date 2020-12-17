import numpy as np

from invador import Invador
from dqn_agent import DQNAgent

if __name__ == "__main__":
    # parameters
    n_epochs = 2000

    # environment, agent
    env = Invador(simple=False)
    agent = DQNAgent(env.enable_actions, [env.IMG_SIZE, env.IMG_SIZE], env.name)
    agent.compile()

    training_log = []
    for e in range(n_epochs):
        # reset
        frame = 0
        loss = 0.0
        Q_max = 0.0
        env.reset()
        state_t_1, reward_t, terminal = env.observe()

        while not terminal:
            state_t = state_t_1

            # execute action in environment
            action_t = agent.select_action(state_t.reshape(env.IMG_SIZE*env.IMG_SIZE), agent.exploration)
            env.step(action_t)

            # observe environment
            state_t_1, reward_t, terminal = env.observe()

            # store experience
            agent.store_experience(state_t.reshape(env.IMG_SIZE*env.IMG_SIZE), action_t, reward_t, state_t_1.reshape(env.IMG_SIZE*env.IMG_SIZE), terminal)

            # experience replay
            agent.experience_replay()

            # for log
            frame += 1
            loss += agent.current_loss
            Q_max += np.max(agent.Q_values(state_t.reshape(env.IMG_SIZE*env.IMG_SIZE)))
            
        REWARD = reward_t
        msg = "EPOCH: {:03d}/{:03d} | REWARD: {:03d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(e, n_epochs - 1, REWARD, loss / frame, Q_max / frame)
        print(msg)
        training_log.append(msg + "\n")

    # save model
    agent.save_model()

    # save log
    with open("./training_log.txt", "w") as f:
        for log in training_log:
            f.write(log)
