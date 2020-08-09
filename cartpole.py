'''
cartpole.py
Joe Nguyen | 05 Aug 2020
'''

import os
import argparse
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


AP = argparse.ArgumentParser()
AP.add_argument(
    '-e', '--episodes', dest='episodes', default='1', help='Number of episodes'
)
AP.add_argument(
    '-m', '--method', dest='method', default='net',
    help='Strategy to choose action',
)
AP.add_argument(
    '-s', '--stop', dest='stop', action='store_true',
    help='Stop simulation at terminal state',
)
AP.add_argument(
    '-v', '--verbose', dest='verbose', action='store_true',
    help='Show action, state, reward, done',
)
AP.add_argument(
    '-mp', '--model-path', dest='path_model',
    default='./models/model.pth',
    help='Model path',
)
AP.set_defaults(stop=True, verbose=False)
args = AP.parse_args()


class Net(nn.Module):
    '''Neural net to learn policy
    - 4 input states
    - 2 output actions (L,R)
    '''

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 16)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=1)


def select_action(state, model, method='good', train_nn=False):
    '''
    Args:
        model (pytorch): NN model
    '''
    if method == 'good':
        action = 0 if state[2] + state[3] < 0 else 1
    elif method == 'simple':
        action = 0 if state[2] < 0 else 1
    elif method == 'random':
        action = np.random.random() < 0.5
    elif method == 'net':
        state = torch.from_numpy(state).float().unsqueeze(0)  # (1x4)
        probs = model(state)
        m = Categorical(probs)
        action = m.sample()
        prob = m.log_prob(action)
        action = action.item()

    if train_nn:
        return action, prob
    else:
        return action


def train_nn(net, optimizer, num_episodes=1000,
             path_model='./models/cartpole_nn.pth'):
    '''Train NN. Loss function is defined as the negative sum of
    rewards, where a reward is the survival duration at each timestep
    mutliplied by the NN output (action probability)
    '''
    last_steps = []

    for episode in range(num_episodes):
        state = env.reset()
        probs = []
        loss = 0

        for t in range(1, env._max_episode_steps + 1):
            action, prob = select_action(
                state, model=net, method='net', train_nn=True)
            probs.append(prob)
            state, _, done, _ = env.step(action)

            if done:
                break

        for i, prob in enumerate(probs):
            loss -= (t - i) * prob

        if episode % 99 == 0:
            print(f'{episode}\t last-step: {t}\t',
                  f'loss: {loss.item(): .2f}')

        # Train
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_steps.append(t)

        # End training
        if (
            len(last_steps) > 10 and
            np.mean(last_steps[-10:]) >= env._max_episode_steps * 0.95
        ):
            print('Model training finished.')

            # Save trained model
            torch.save(net.state_dict(), path_model)
            return


def simulate(num_episodes=5, num_steps=None, stop=True, verbose=False):
    '''Cartpole learning simulation
    - Environment's state is describe by a 4-tuple
    (
        x position of cart,
        x velocity of cart,
        angular position of pole,
        angular velocity of pole
    )
    '''
    t_ls = []
    if num_steps is None:
        num_steps = env._max_episode_steps

    for episode in range(num_episodes):

        state = env.reset()

        for t in range(1, num_steps + 1):
            env.render()
            action = select_action(state, model=net, method=args.method)
            # action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            if verbose:
                print(
                    f'{t}: action: {action} | state: {state} |',
                    f'reward: {reward} | done: {done} | info: {info}',
                )
            if done:
                if verbose:
                    print(f'Episode finished after {t + 1} timesteps')
                if stop:
                    break

            t_ls.append(t)
    env.close()

    # Survival score
    score = t_ls[-1] / num_steps
    print(f'Survival score: {score}')
    # score = sum(t_ls) / (len(t_ls) * num_steps)
    # print(f'Survival score: {score} | {sum(t_ls)} {t_ls[-1]} {len(t_ls)}')


# env = gym.make('CartPole-v0')  # 200-step episode
env = gym.make('CartPole-v1')  # 500-step episode
print(f'Environment max episode steps: {env._max_episode_steps}')

# State space and action space
print(f'State space: {env.observation_space}')
print(f'Action space: {env.action_space}')

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.01)


def main(args):

    # Load or train nn model
    if os.path.exists(args.path_model):
        print('Model found...loading')
        net.load_state_dict(torch.load(args.path_model))
    else:
        print('Model training...')
        train_nn(net, optimizer, path_model=args.path_model)

    simulate(
        num_episodes=np.int(args.episodes),
        num_steps=None,
        stop=args.stop,
        verbose=args.verbose
    )


if __name__ == '__main__':
    print(args)
    main(args)
