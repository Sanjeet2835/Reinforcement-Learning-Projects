# 🐍 Snake Game AI - Deep Q-Learning (DQN)

This repository contains a Reinforcement Learning project where an AI agent learns to play the classic Snake game from scratch using **Deep Q-Learning (DQN)**. 

The project is built using Python, Pygame for the environment, and PyTorch for the neural network.

## 🧠 How it Works

The agent has no prior knowledge of the game rules. It learns purely through trial and error using a Reward system:
* **+10 Reward:** Eating the food.
* **-10 Penalty:** Crashing into a wall or itself.
* **0 Reward:** Normal movement.

The AI uses a **Neural Network** to predict the best action based on the current state of the game (danger proximity, current direction, and food location). It uses **Experience Replay** to store past moves in memory and trains on batches of these memories to improve its decision-making over time.

## 📂 Project Structure

* `snake_game_ai.py`: The Pygame environment. It handles the game logic, UI rendering, and calculates the 7-value state array.
* `agent.py`: The RL agent. It handles the epsilon-greedy exploration strategy and stores the experience memory buffer (the "flashcards").
* `model.py`: The PyTorch Neural Network architecture (Linear Q-Net) and the Q-Trainer that updates the network weights using the Bellman Equation.
* `train.py`: The main training loop. It connects the agent and the game, running infinite episodes to train the model.
* `play.py`: The inference script. It loads a pre-trained model and lets you watch the AI play without any randomness or training overhead.
* `model/`: Directory where the trained PyTorch model (`model.pth`) is saved.

## 🚀 How to Run

### 1. Install Dependencies
Make sure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```
*(Requirements should include `pygame`, `torch`, `numpy`)*

### 2. Train the AI from Scratch
To watch the AI learn from completely random movements to a master snake, run the training script:
```bash
python train.py
```
*Note: Every time the AI beats its high score, it automatically saves/overwrites its brain to `model/model.pth`.*

### 3. Watch the Trained AI Play (Inference)
If you just want to see the AI play using the pre-trained model (no exploration, pure exploitation), run:
```bash
python play.py
```
Note: 

## 🛠️ Concepts Covered
* Reinforcement Learning (RL)
* Deep Q-Networks (DQN)
* The Bellman Equation
* Epsilon-Greedy Exploration Strategy
* Experience Replay (Long-term & Short-term memory)
* PyTorch Model Checkpointing
