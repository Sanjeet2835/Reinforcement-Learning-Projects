import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os # 

# 1. THE BRAIN
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Here we define our layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # This is how the state flows through the brain to get the Q-values
        x = F.relu(self.linear1(x)) # ReLU is an activation function (adds non-linearity)
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        # Create a folder called 'model' if it doesn't exist
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        # Save the brain's weights to a file!
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# 2. THE TRAINER
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # The optimizer is the tool that actually updates the weights of the brain
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # The criterion is how we measure our errors (Mean Squared Error)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert our arrays into PyTorch Tensors (PyTorch's version of arrays)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # 1. Get current Q values predicted by our model
        pred = self.model(state)

        # 2. Calculate the "Target" Q values using the Bellman equation
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Q_new = reward + gamma * max(next_predicted Q value)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # Update the specific action we took with the new Q value
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 3. Update the neural network weights
        self.optimizer.zero_grad() # Clear old calculations
        loss = self.criterion(target, pred) # Calculate the error
        loss.backward() # Backpropagation (The magic of machine learning)
        self.optimizer.step() # Tweak the weights