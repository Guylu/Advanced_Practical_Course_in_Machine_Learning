import torch
import torch.nn.functional as F
from memory import ReplayMemory, Transition
import random
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
import gym
import numpy as np
import models


class QPolicy(BasePolicy):
    # partial code for q-learning
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(QPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)
        self.q_table = np.zeros((1, 9, 9, 10, 3))
        self.lr = lr
        self.steps = 0
        self.policy_net = models.SimpleModel(9 * 10, 9 * 10, 3)
        self.target_net = models.SimpleModel(9 * 10, 9 * 10, 3)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state, epsilon, global_step=None):
        """
        select an action to play
        :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        :param epsilon: epsilon...
        :param global_step: used for tensorboard logging
        :return: return single action as integer (0, 1 or 2).
        """
        self.steps += 1
        random_number = random.random()
        if random_number > epsilon:
            # here you do the action-selection magic!
            # todo: YOUR CODE HERE
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return int(self.model(state).max(1)[1].view(1, 1))
        else:
            return self.action_space.sample()  # return action randomly

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
            next_state_batch = torch.cat(batch.next_state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            reg = 3

            if reg == 1:
                reward_batch /= 5
            elif reg == 2:
                reward_batch = (reward_batch - np.mean(reward_batch.detach().numpy())) / np.std(
                    reward_batch.detach().numpy())
            elif reg == 3:
                reward_batch = (reward_batch - np.min(reward_batch.detach().numpy())) / (
                            np.max(reward_batch.detach().numpy()) - np.min(reward_batch.detach().numpy()))

            # do your optimization magic here!
            self.optimizer.zero_grad()
            res = self.model(state_batch)
            res_next = torch.max(self.model(next_state_batch), dim=1)[0]
            idx = action_batch.reshape(-1, 1).long()
            with torch.no_grad():
                no_grad_calc = (reward_batch + self.gamma * res_next)
            res_at_idx = torch.gather(res, 1, idx)
            loss = torch.mean((res_at_idx - no_grad_calc) ** 2)
            # torch.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
