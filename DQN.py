import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

replay_buffer_capacity = int(1e6)


# TODO replace with dynamic num hidden layers
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        output = F.leaky_relu(self.layer1(x))
        output = F.leaky_relu(self.layer2(output))
        output = self.layer3(output)
        return output


class ConvFC(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, input_size, hidden_size, output_size, kernel_size=3):
        super(ConvFC, self).__init__()
        self.conv_out_channels = conv_out_channels
        self.layer1 = nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=kernel_size)
        self.conv_result_size = (input_size - kernel_size + 1) # no stride or pad here
        self.fc_size = self.conv_result_size ** 2 * self.conv_out_channels
        self.layer2 = nn.Linear(self.fc_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        assert len(x.shape) >= 3
        # print(x.shape)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        conv_output = F.leaky_relu(self.layer1(x))
        # print(conv_output)
        output = conv_output.reshape(-1, self.fc_size)
        # print(output)
        output = F.leaky_relu(self.layer2(output))
        # print(output)
        output = self.layer3(output)
        # print(output)

        return output


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition_tuple):
        """Saves a transition."""
        self.memory.append(transition_tuple)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:

    def __init__(self, action_space_low, action_space_high, neural_net, replay_buffer=None, batch_size=100, tau=3e-3,
                 eps_start=0.9, eps_end=0.1, eps_decay=0.995, lr=0.005, gamma=1.00, episode_reward_history_len=1000):
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high # inclusive
        self.gamma = gamma
        self.lr = lr
        self.q_being_updated = copy.deepcopy(neural_net)
        self.q_target_net = copy.deepcopy(neural_net)
        if replay_buffer is None:
            self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        else:
            self.replay_buffer = replay_buffer
        # self.training_reset_steps = training_reset_steps
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.batch_size = batch_size
        # self.optimizer = torch.optim.Adam(self.q_being_updated.parameters(),
        #                                   lr=self.lr)
        # SGD optim seems to help convergence to TFT in a TFT + defect pool. Probably something
        # weird going on with Adam - or maybe momentum is a bad idea when opponents change
        # relatively frequently
        self.optimizer = torch.optim.Adam(self.q_being_updated.parameters(), self.lr)
        self.reward_total = 0
        self.episode_reward = 0
        self.episode_reward_history = deque(maxlen=episode_reward_history_len)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def to(self, device):
        self.q_being_updated.to(device)
        self.q_target_net.to(device)


    def train_nn(self, discount, batch_size):

        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        batch = np.array(batch)

        states = np.stack(batch[:,0])
        actions = torch.Tensor(np.array(batch[:,1], dtype=np.float))
        rewards = torch.Tensor(np.array(batch[:,2], dtype=np.float))
        next_states = np.stack(batch[:,3])
        dones = torch.Tensor((np.array(batch[:,4], dtype=np.float)))

        predicted_Q = self.Q(states, self.q_being_updated)

        reshaped_predicted_Q = predicted_Q.gather(1, actions.long().view(-1,1))

        next_states_maxQ = self.maxQ(next_states, self.q_target_net)

        next_states_maxQ = next_states_maxQ[0]

        # print(predicted_Q)
        targets = (rewards + discount * next_states_maxQ * (1-dones))
        # print(targets)

        targets = targets.view(-1, 1)
        loss = F.mse_loss(reshaped_predicted_Q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_target_net, self.q_being_updated)


    def soft_update(self, target_net, curr_net):
        for target_param, curr_param in zip(target_net.parameters(), curr_net.parameters()):
            target_param.data.copy_(self.tau * curr_param.data + (1-self.tau) * target_param.data)

    def push_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push((state, action, reward, next_state, done))

    def q_learn_update(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # self.replay_buffer.push((state, action, reward, next_state, done))
        self.train_nn(self.gamma, batch_size)
        self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)



    def Q(self, states, neural_net_to_use, no_grad=False, print_act=False, normalize_rgb=True):
        """This function returns, for a set of states as input, a set of
        sets of Q values as output, where each subset contains the Q values
        across all possible actions. Then we use max or argmax to extract
        either the desired Q value or the desired action to take
        """

        states = np.array(states)
        states = torch.from_numpy(states)
        states = states.float()

        # print(states)

        # print(neural_net_to_use.layer1.weight)

        states = states.to(device=self.device)

        if normalize_rgb:
            states = states / 255 # just scale to 0-1 for now.

        if no_grad:
            with torch.no_grad():
                output = neural_net_to_use(states)
        else:
            output = neural_net_to_use(states)

        if print_act:
            print(output.cpu())

        return output.cpu()

    def maxQ(self, states, neural_net_to_use):
        # The no_grad=True line below is very important on this part!
        # Why? Well when we backprop, we only want the forward pass to count
        # towards the autogradient/diff once
        maxQ = self.Q(states, neural_net_to_use, no_grad=True)
        maxQ = torch.max(maxQ, dim=1)

        return maxQ

    def act(self, state, epsilon=None, print_act=False):
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy action
        if np.random.random() < epsilon:
            return np.random.randint(self.action_space_low, self.action_space_high + 1)

        best_action = torch.argmax(self.Q(state, self.q_being_updated, no_grad=True, print_act=print_act))

        return best_action.item()
