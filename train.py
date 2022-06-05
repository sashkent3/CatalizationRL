import math
import random
from collections import deque, namedtuple
from functools import partial, reduce
from itertools import count, repeat
from json import dumps
from operator import is_not, mul

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda:0'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'next_actions', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w):
        super(DQN, self).__init__()
        embedding_dim = 3
        self.board_embedding = nn.Embedding(
            num_embeddings=13, embedding_dim=embedding_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(embedding_dim + 3, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
        )

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def pool_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = pool_size_out(conv_size_out(pool_size_out(conv_size_out(w))))
        convh = pool_size_out(conv_size_out(pool_size_out(conv_size_out(h))))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, board, place):
        place = place.to(device)
        board = board.to(device=device, dtype=torch.int)
        board = self.board_embedding(board).permute(0, 3, 2, 1)
        x = torch.cat((place, board), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.head(x)
        from itertools import product


BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 10000 * 50
TARGET_UPDATE = 10
LEARNING_RATE = 1e-4

screen_height, screen_width = 17, 18
action_space_shape = (2, screen_width * screen_height)
n_actions = reduce(mul, action_space_shape, 1)

policy_net = DQN(screen_height, screen_width).to(device)
target_net = DQN(screen_height, screen_width).to(device)
target_net.load_state_dict(state_dict=policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(10000)
criterion = nn.MSELoss()

steps_done = 0


def select_action(board, available_actions, available_tiles):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * (steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        actions = []
        place_batch = torch.zeros(
            (len(set(available_tiles)) * len(available_actions['place']), 3, screen_height, screen_width), dtype=torch.bool)
        for place_idx in range(len(available_actions['place'])):
            place_batch[place_idx, available_tiles[0], available_actions['place']
                        [place_idx][1], available_actions['place'][place_idx][0]] = 1
            actions.append(('place', available_actions['place'][place_idx], 1))
        if available_tiles[0] != available_tiles[1]:
            for place_idx in range(len(available_actions['place'])):
                place_batch[len(available_actions['place']) + place_idx, available_tiles[1],
                            available_actions['place'][place_idx][1], available_actions['place'][place_idx][0]] = 1
                actions.append(
                    ('place', available_actions['place'][place_idx], 2))
        state_actions = board.expand(place_batch.shape[0], -1, -1)
        with torch.no_grad():
            best_action_idx = policy_net(board.expand(
                place_batch.shape[0], -1, -1), place_batch).argmax()
        return actions[best_action_idx], place_batch[best_action_idx]

    else:
        tile_idx = random.randint(0, 1)
        place = random.choice(available_actions['place'])
        action_tensor = torch.zeros(
            3, screen_height, screen_width, dtype=torch.bool)
        action_tensor[available_tiles[tile_idx], place[1], place[0]] = 1
        return ('place', place, tile_idx), action_tensor


episode_durations = []
episode_losses = []
episode_rewards = []

is_not_none = partial(is_not, None)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(map(is_not_none, batch.next_state)), device=device, dtype=torch.bool)
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.stack(batch.action).to(device)
    reward_batch = torch.stack(batch.reward).to(device)
    policy_net.train()
    state_action_values = policy_net(state_batch, action_batch)
    next_actions_shapes = [t.shape[0]
                           for t in batch.next_actions if t is not None]
    next_actions_batch = torch.cat(
        tuple(filter(is_not_none, batch.next_actions)))
    next_state_batch = torch.cat(tuple(map(lambda state, size: state.expand(
        size, -1, -1), filter(is_not_none, batch.next_state), next_actions_shapes)))
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = torch.tensor(tuple(map(lambda t: t.max().detach(
        ), target_net(next_state_batch, next_actions_batch).split(next_actions_shapes))), device=device)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(dim=1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


num_episodes = 10000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    requests.post('http://localhost:5757/reset_game', timeout=5)
    state = requests.get(
        'http://localhost:5757/get_state', timeout=5).json()
    available_tiles = state['available_tiles']
    board = torch.tensor(state['board'], dtype=torch.uint8)
    episode_reward = 0
    available_actions = requests.get(
        'http://localhost:5757/get_moves', timeout=5).json()
    losses = []
    for t in count():
        # Select and perform an action
        if available_actions['heal_upgrade']:
            requests.post('http://localhost:5757/make_move',
                          data=dumps(('heal_upgrade',)), timeout=5)
        if available_actions['freeze']:
            requests.post('http://localhost:5757/make_move',
                          data=dumps(('freeze',)), timeout=5)
        if len(available_actions['heal']) > 0:
            requests.post('http://localhost:5757/make_move', data=dumps(('heal',
                                                                         random.choice(available_actions['heal']))), timeout=5)
        action, action_tensor = select_action(
            board, available_actions, available_tiles)
        reward = 0
        r = requests.post('http://localhost:5757/make_move',
                          data=dumps(action), timeout=5).json()
        won = r.get('won', None)
        reward += r['reward']
        if won is None:
            done = False
            reward += 10
        else:
            done = True
            if won:
                reward += 1000
            else:
                reward -= 1000
        episode_reward += reward
        reward = torch.tensor(reward, dtype=torch.int16, device=device)
        if not done:
            state = requests.get(
                'http://localhost:5757/get_state', timeout=5).json()
            available_actions = requests.get(
                'http://localhost:5757/get_moves', timeout=5).json()
            available_tiles = state['available_tiles']
            board = torch.tensor(state['board'], dtype=torch.uint8)
            next_actions = torch.zeros(
                (len(set(available_tiles)) * len(available_actions['place']), 3, screen_height, screen_width), dtype=torch.bool)
            for place_idx in range(len(available_actions['place'])):
                next_actions[place_idx, available_tiles[0], available_actions['place']
                             [place_idx][1], available_actions['place'][place_idx][0]] = 1
            if available_tiles[0] != available_tiles[1]:
                for place_idx in range(len(available_actions['place'])):
                    next_actions[len(available_actions['place']) + place_idx, available_tiles[1],
                                 available_actions['place'][place_idx][1], available_actions['place'][place_idx][0]] = 1
            next_state = board.expand(next_actions.shape[0], -1, -1)
        else:
            next_state = None
            next_actions = None

        # Store the transition in memory
        memory.push(board, action_tensor,
                    next_state, next_actions, reward)

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()
        if loss is not None:
            losses.append(loss)
        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(episode_reward)
            break
    episode_losses.append(np.mean(losses))
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        print('episode', i_episode)
        print(eps_threshold)
        if len(episode_rewards[-TARGET_UPDATE:]) > 0:
            print(np.mean(episode_rewards[-TARGET_UPDATE:]))
        target_net.load_state_dict(policy_net.state_dict())

torch.save(policy_net.state_dict(), "policy_net.pt")
