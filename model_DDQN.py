import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss().to(device)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
  
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0).to(device)
            next_state = torch.unsqueeze(next_state, 0).to(device)
            action = torch.unsqueeze(action, 0).to(device)
            reward = torch.unsqueeze(reward, 0).to(device)
            done = (done, )

        pred = self.model(state).to(device)
        
        target = pred.clone()
        
        next_state_actions = self.model(next_state).detach()
        next_state_values = self.target_model(next_state).detach()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                best_action = torch.argmax(next_state_actions[idx])
                Q_new = reward[idx] + self.gamma * next_state_values[idx][best_action]

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())



