import torch
import torch.nn.functional as F
from memory import ReplayMemory, Transition
import random
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
import gym
import numpy as np
import models
from scipy.stats import entropy


class van_rein(BasePolicy):
    # partial code for q-learning
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(van_rein, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)
        self.lr = lr
        self.steps = 0
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state, epsilon, global_step=None):
        """
        select an action to play
        :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        :param epsilon: epsilon...
        :param global_step: used for tensorboard logging
        :return: return single action as integer (0, 1 or 2).
        """
        self.steps += 1

        # here you do the action-selection magic!
        # todo: YOUR CODE HERE

        self.model.eval()
        s = self.model(state)
        cats = torch.distributions.Categorical(s)
        a = cats.sample()
        return int(a)

    def optimize(self, batch_size, global_step=None):
        # optimize your model

        if len(self.memory) < batch_size:
            return None

        self.memory.batch_size = batch_size
        for transitions_batch in self.memory:
            # transform list of tuples into a tuple of lists.
            # explanation here: https://stackoverflow.com/a/19343/3343043
            batch = Transition(*zip(*transitions_batch))

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # self.optimizer.zero_grad()

            r = reward_batch.detach().numpy()

            reg = 3

            if reg == 1:
                r /= 5
            elif reg == 2:
                r = (r - np.nanmean(r)) / (np.nanstd(r)+0.0000001)
            elif reg == 3:
                r = (r - np.nanmin(r)) / (np.nanmax(r) - np.nanmin(r)+0.0000001)

            size = r.size

            pows = np.tile(np.arange(size), (size, 1)) - np.arange(size).reshape(-1, 1)
            pos_pows = np.triu(pows)
            gammas = np.triu(np.power(self.gamma, pos_pows))
            R = np.sum(gammas * r, axis=1)
            R_t = torch.from_numpy(R).reshape(-1, 1)

            res = self.model(state_batch)

            a = 0.01
            e = a * torch.distributions.Categorical(probs=res).entropy()

            obj = torch.log(torch.gather(res, 1, action_batch.reshape(-1, 1).long())) * (R_t - e)
            obj2 = torch.mean(obj)

            loss = -obj2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
