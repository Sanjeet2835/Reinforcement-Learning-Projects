import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # Experience Replay Buffer
        
        # Initialize our Neural Network (7 inputs, 256 hidden nodes, 3 outputs)
        self.model = Linear_QNet(7, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Grab a random batch of 1000 memories
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory

        # Extract states, actions, rewards, etc., from the batch
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train on just the single step we just took
        self.trainer.train_step((state,), (action,), (reward,), (next_state,), (done,))

    def get_action(self, state):
        # Epsilon-greedy: Decreases random moves as n_games increases
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Let the Neural Network predict the best move!
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move