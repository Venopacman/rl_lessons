import gym.spaces
import matplotlib.pyplot as plt
import numpy as np
from qlearning_template import QLearningAgent
from sarsa_template import SarsaAgent
from cliff_walking import CliffWalkingEnv


def play_and_train(env, agent, t_max=10 ** 4):
    """ This function should
    - run a full game (for t_max steps), actions given by agent
    - train agent whenever possible
    - return total reward
    """
    total_reward = 0.0
    # Play & train game
    s = env.reset()
    for it in range(t_max):
        a = agent.get_action(s)
        new_s, r, is_done, _ = env.step(a)
        # Update rewards
        agent.update(s, a, new_s, r)
        total_reward += r
        s = new_s
        if is_done:
            break
    return total_reward

def play_and_train_s(env, agent, t_max=10 ** 4):
    """ This function should
    - run a full game (for t_max steps), actions given by agent
    - train agent whenever possible
    - return total reward
    """
    total_reward = 0.0
    # Play & train game
    s = env.reset()
    for it in range(t_max):
        a = agent.get_action(s)
        new_s, r, is_done, _ = env.step(a)
        # Update rewards
        agent.update(s, a, new_s, r, agent.get_action(new_s))
        total_reward += r
        s = new_s
        if is_done:
            break
    return total_reward


if __name__ == '__main__':
    max_iterations = 1000
    visualize = True
    # Create Taxi-v2 env
    env = CliffWalkingEnv()

    n_states = env.nS
    n_actions = env.nA

    print('States number = %i, Actions number = %i' % (n_states, n_actions))

    # create q learning agent with
    alpha = 0.1
    get_legal_actions = lambda s: range(n_actions)
    epsilon = 0.5
    discount = 0.99

    agent = QLearningAgent(alpha, epsilon, discount, get_legal_actions)
    s_agent = SarsaAgent(alpha, epsilon, discount, get_legal_actions)

    plt.figure(figsize=[10, 4])
    rewards = []
    s_rewards = []
    s = env.reset()
    # env.render()
    # Training loop
    for i in range(max_iterations):

        rewards.append(play_and_train(env, agent))
        s_rewards.append(play_and_train_s(env, s_agent))
        # Decay agent epsilon
        agent.epsilon = agent.epsilon * discount
        s_agent.epsilon = s_agent.epsilon * discount
        if i % 100 == 0:
            print('Iteration {}, Average reward delta {:.2f}, Epsilon {:.3f}'
                  .format(i, np.mean(s_rewards[-100:]) - np.mean(rewards[-100:]), agent.epsilon))

        if visualize and (i % 100 == 0):
            plt.subplot(1, 2, 1)
            plt.plot(rewards, color='r')
            plt.plot(s_rewards, color='b')
            plt.xlabel('Iterations')
            plt.ylabel('Total Reward')

            plt.subplot(1, 2, 2)
            plt.hist(rewards, bins=20, range=[-700, +20], color='blue', label='Rewards distribution')
            plt.hist(s_rewards, bins=20, range=[-700, +20], color='yellow', label='Rewards distribution')
            plt.xlabel('Reward')
            plt.ylabel('p(Reward)')
            plt.draw()
            plt.pause(0.5)
            plt.cla()
    print(' Final Average Sarsa {:.2f} and Q_learning {:.2f} rewards'
          .format(np.mean(s_rewards[-50:]), np.mean(rewards[-50:])))
