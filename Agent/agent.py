import torch
import torch.nn as nn
import torch.optim as optim
import Agent.memory_utils as memutils


class DQNAgent:
    def __init__(self, network, config_dict):
        self._replay_memory = memutils.ReplayMemory(config_dict['replay_size'])
        self._config_dict = config_dict
        self._network = network().to('cuda')
        self._target_network = network().to('cuda')

        if config_dict['loss_type'] == 'huber':
            self._loss = nn.SmoothL1Loss()
        elif config_dict['loss_type'] == 'mse':
            self._loss = nn.MSELoss()
        else:
            raise ValueError("config_dict['loss_type'] should be 'huber' or 'mse'.")
        
        self._optimizer = optim.RMSprop(self._network.parameters(), config_dict['lr'])

        self.sync_network()
        self._target_network.eval()

    def _get_loss(self, state, action, reward, next_state, done):
        masked_next_reward = (1 - done) * torch.max(self._target_network.forward(next_state), dim=1)[0]
        y_target = reward + (masked_next_reward * self._config_dict['discount_factor'])
        y = torch.squeeze(self._network(state).gather(1, action.unsqueeze(1)))
        loss = self._loss(y, y_target)
        return loss

    def _optimize(self, loss):
        self._optimizer.zero_grad()
        loss.backward()

        for var in self._network.parameters():
            torch.clamp_(var.grad.data, -self._config_dict['gradient_clip'], self._config_dict['gradient_clip'])

        self._optimizer.step()

    @staticmethod
    def _batch2tensor(batch):
        state = torch.Tensor(batch[0]).cuda()
        action = torch.Tensor(batch[1]).cuda().long()
        reward = torch.Tensor(batch[2]).cuda()
        next_state = torch.Tensor(batch[3]).cuda()
        done = torch.Tensor(batch[4]).cuda()
        return state, action, reward, next_state, done

    def sync_network(self):
        self._target_network.load_state_dict(self._network.state_dict())

    def get_action(self, state):
        with torch.no_grad():
            self._network.eval()
            q = self._network.forward(torch.Tensor([state]).cuda()).to('cuda')
            action = torch.argmax(q, dim=1).cpu().numpy()
            self._network.train()
            return action[0]

    def append_memory(self, state, action, reward, next_state, done):
        self._replay_memory.append(state, action, reward, next_state, done)

    def train(self):
        batch = self._replay_memory.get_batch(self._config_dict['batch_size'])
        state, action, reward, next_state, done = self._batch2tensor(batch)
        loss = self._get_loss(state, action, reward, next_state, done)
        self._optimize(loss)
        return loss

    def save_weights(self, path):
        torch.save(self._network.state_dict(), path)

    def restore_weights(self, path):
        state_dict = torch.load(path)
        self._network.load_state_dict(state_dict)
        self.sync_network()
