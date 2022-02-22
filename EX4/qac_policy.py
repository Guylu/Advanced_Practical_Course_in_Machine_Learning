import torch
import torch.nn.functional as F
from memory import ReplayMemory, Transition
import random
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
import gym
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical
import numpy as np


class QACPolicy(BasePolicy):
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(QACPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)

        self.P, self.Q = model
        self.opt_P = optim.RMSprop(self.P.parameters(), lr=5e-5)
        self.opt_Q = optim.Adam(self.Q.parameters())
        self.writer = summery_writer
        self.train_iter = 0

    def select_action(self, state, epsilon, global_step=None):
        """
        select an action to play
        :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        :param epsilon: epsilon...
        :param global_step: used for tensorboard logging
        :return: return single action as integer (0, 1 or 2).
        """
        self.P.eval()
        output = self.P(state)
        cats = Categorical(output)
        action = cats.sample()
        return action


    def optimize(self, batch_size, global_step=None):
        # optimize your model

        if len(self.memory) < batch_size:
            return None

        self.memory.batch_size = batch_size
        running_loss = 0
        for transitions_batch in self.memory:
            self.opt_P.zero_grad()
            self.opt_Q.zero_grad()

            # transform list of tuples into a tuple of lists.
            # explanation here: https://stackoverflow.com/a/19343/3343043
            batch = Transition(*zip(*transitions_batch))

            state_batch = torch.cat(batch.state)
            next_state_batch = torch.cat(batch.next_state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)


            r = reward_batch
            # normalize the reward

            # normalization
            norm_type = 3

            if norm_type == 0:
                pass
            if norm_type == 1:
                r /= 5
            elif norm_type == 2:
                r = (r - r.mean()) / (r.std() + 0.0000000001)
            elif norm_type == 3:
                r = (r - r.min()) / (r.max() - r.min() + 0.000000001)

            # Training the actor
            with torch.no_grad():
                idx = action_batch.reshape(-1, 1).long()
                outputs = self.Q(state_batch)
                Q_vals = torch.gather(outputs, 1, idx)

            probs = self.P(state_batch)
            idx = action_batch.reshape(-1, 1).long()

            objs = torch.log(torch.gather(probs, 1, idx)) * Q_vals
            obj = -torch.mean(objs)

            # make a gradient step
            obj.backward()

            max_norm = 1
            nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm)

            self.opt_P.step()

            # Train the critic
            self.opt_Q.zero_grad()

            with torch.no_grad ():
                # make a Q target batch of size batch_size - 1
                Q_next = self.Q(next_state_batch)[:-1]
                idx = action_batch.long()
                idx = torch.roll(idx, -1)[:-1]
                idx = idx.reshape(-1, 1)
                Q_next_val = torch.gather(Q_next, 1, idx)
                Q_target = r[:-1].reshape(-1, 1) + self.gamma * Q_next_val

            idx = action_batch.reshape(-1, 1).long()[:-1]
            outputs = self.Q(state_batch)[:-1]
            Q_vals = torch.gather(outputs, 1, idx)

            mse_loss = nn.MSELoss()
            loss = mse_loss(Q_vals, Q_target.reshape(-1, 1))
            running_loss += loss.item()


            # make a gradient step
            loss.backward()

            self.print_grads(self.Q)

            self.opt_Q.step()


        self.writer.add_scalar('training/loss', running_loss, self.train_iter)
        self.train_iter += 1

    def print_grads(self, model):
        # and log total norm of grads
        grads = []
        total_norm = 0
        for param in model.parameters():
            grads.append(param.grad.view(-1))
            total_norm += param.grad.data.norm(2).item()
        grads = torch.cat(grads)
        self.writer.add_scalar('training/grads_norm', total_norm, self.train_iter)
        # print(grads.shape)
        # print(grads[0:15])

