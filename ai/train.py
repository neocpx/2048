import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
import random
import math
from itertools import count

commands={0:'left',1:'right',2:'up',3:'down'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define the 2048 environment
import random

class Game2048:
    def __init__(self):
        self.state = [[0] * 4 for _ in range(4)]
        self.setNumber(2)
        self.score = 0

    def reset(self):
        self.__init__()

    def setNumber(self, n=1):
        for _ in range(n):
            emptyPos = [(i, j) for i in range(4) for j in range(4) if self.state[i][j] == 0]
            if not emptyPos:
                return True
            randPos = random.choice(emptyPos)
            self.state[randPos[0]][randPos[1]] = random.choice((2, 4))
        return False

    def compact_and_merge(self, line):
        compacted_line = [tile for tile in line if tile != 0]
        merged_line = []
        i = 0
        while i < len(compacted_line):
            if i < len(compacted_line) - 1 and compacted_line[i] == compacted_line[i + 1]:
                merged_line.append(compacted_line[i] * 2)
                self.score += compacted_line[i] * 2  # Update score when merging
                i += 2
            else:
                merged_line.append(compacted_line[i])
                i += 1
        # Fill the remaining spaces with zeros
        merged_line += [0] * (len(line) - len(merged_line))
        return merged_line

    def move(self, command):
        newState = [r[:] for r in self.state]
        for idx, row_or_column in enumerate(newState):
            if command == 'left':
                newState[idx] = self.compact_and_merge(row_or_column)
            elif command == 'right':
                newState[idx] = self.compact_and_merge(row_or_column[::-1])[::-1]
            elif command == 'up':
                column = self.compact_and_merge([newState[row][idx] for row in range(len(newState))])
                for row in range(len(newState)):
                    newState[row][idx] = column[row]
            elif command == 'down':
                column = self.compact_and_merge([newState[row][idx] for row in range(len(newState))][::-1])[::-1]
                for row in range(len(newState)):
                    newState[row][idx] = column[row]
        self.state=newState
        return newState

    def updateScore(self):
        # Clear the current score before updating
        self.score = 0
        # Update score for all tiles on the board
        self.score += sum(tile_value for row in self.state for tile_value in row if tile_value not in (0, 2))

    def is_game_over(self):
        # Check if there are any empty spaces
        if any(tile == 0 for row in self.state for tile in row):
            return False

        # Check if there are any adjacent tiles with the same value
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == self.state[i + 1][j] or self.state[i][j] == self.state[i][j + 1]:
                    return False

        # Check the last row and last column
        for i in range(3):
            if self.state[3][i] == self.state[3][i + 1] or self.state[i][3] == self.state[i + 1][3]:
                return False

        return True


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


# Get number of actions from gym action space
n_actions = len(commands)
# Get the number of state observations
game=Game2048()
n_observations = len(game.state)*len(game.state[0])

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    q_values = policy_net(state)
    if sample > eps_threshold:
        with torch.no_grad():
            # max(1) gets the maximum Q-value along dimension 2 (actions)
            # indices contains the index of the action with the maximum Q-value for each position
            selected_action_indices = q_values.max(1).indices
            return selected_action_indices.view(1,1)
    else:
        return torch.tensor([[random.choice((0, 1, 2, 3))]], device=device, dtype=torch.long)


episode_durations = []
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for    
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch sstate according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 50

save_path = './ai/models/trained_model.pth'

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    game.reset()
    state=game.state
    state = torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation = game.move(commands[action.item()])
        game.updateScore()
        reward = game.score
        terminated = game.is_game_over()
        reward = torch.tensor([reward], device=device)
        game.setNumber()
        done = terminated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).flatten().unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            print(f'\n score: game.score')
            episode_durations.append(t + 1)
            if (i_episode + 1) % 100 == 0:
                save_episode = i_episode + 1
                current_save_path = save_path.format(save_episode)
                torch.save({
                    'policy_net_state_dict': policy_net.state_dict(),
                    'target_net_state_dict': target_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_durations': episode_durations
                }, current_save_path)

                print(f'Model saved at: {current_save_path}')
            break
    
    progress = (i_episode + 1) / num_episodes
    bar_length = 20
    bar = '=' * int(bar_length * progress) + '-' * (bar_length - int(bar_length * progress))
    print(f'\rEpisode [{i_episode + 1}/{num_episodes}] Progress: [{bar}] {progress * 100:.2f}%', end='', flush=True)


print('\nComplete')