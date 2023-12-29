import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        x = x.to(device)
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = ConvBlock(16, 256)
        self.conv2 = ConvBlock(256, 256)
        self.conv3 = ConvBlock(256, 256)
        self.dense1 = nn.Linear(256 * 16, 128)
        self.dense6 = nn.Linear(128, 4)
    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.dropout(self.dense1(x))
        return self.dense6(x)

def encode_state(board):
  board_flat = [0 if e == 0 else int(math.log(e, 2)) for e in board.flatten()]
  board_flat = torch.LongTensor(board_flat)
  board_flat = F.one_hot(board_flat, num_classes=16).float().flatten()
  board_flat = board_flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)
  return board_flat


class DQNAgent:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()

    def select_best_move(self, state):
        with torch.no_grad():
            state=np.array(state)
            state = encode_state(state).float().to(self.device)
            action = self.policy_net(state).max(1)[1].view(1, 1)
            return action.item()

def generate_2048_state():
    # Initialize the game board
    board = np.zeros((4, 4), dtype=int)
    
    # Add two initial cells with a value of 2 or 4
    for _ in range(2):
        i, j = (board == 0).nonzero()
        if i.size != 0:
            rnd = random.randint(0, i.size - 1)
            board[i[rnd], j[rnd]] = 2 * ((random.random() > .9) + 1)
    
    return board

def test_dqn_agent_inference_speed(agent, num_iterations=1000):
    start_time = time.time()

    for _ in range(num_iterations):
        state = generate_2048_state()
        action = agent.select_best_move(state)

    end_time = time.time()
    elapsed_time = end_time - start_time
    average_time = elapsed_time / num_iterations

    print(f"Average time per inference: {average_time:.6f} seconds")

def main():
    # Replace 'your_model_path.pth' with the actual path to your saved model file
    model_path = './ai/models/policy_net.pth'
    print("loaded")
    agent = DQNAgent(model_path)

    # Test inference speed
    test_dqn_agent_inference_speed(agent)

if __name__ == "__main__":
    main()