from . import BasePolicy
import gym

import numpy as np

class QBasePolicy(BasePolicy):
    def __init__(self,
                 action_space,
                 nS,
                 discount_rate=0.9,
                 learning_rate=0.8,
                 explore_rate_max=0.9,
                 explore_rate_min=0.01,
                 explore_rate_decay=0.005):
        super().__init__()
        self.dr = discount_rate
        self.lr = learning_rate
        self.er = explore_rate_max
        self.er_max = explore_rate_max
        self.er_min = explore_rate_min
        self.er_decay = explore_rate_decay

        if not isinstance(nS, np.ndarray):
            self.nS = np.array(nS)
        else:
            self.nS = nS

        if not isinstance(action_space,gym.spaces.MultiDiscrete):
            self.nA = np.array(action_space.n)

    def max_er(self, er_max=None):
        if er_max is not None:
            self.er_max = er_max
        self.er = self.er_max
        return self.er

    def zero_er(self):
        self.er = 0.0
        return self.er

    def decay_er(self, step):
        self.er = self.er_min + (
                self.er_max - self.er_min) * np.exp(
            -self.er_decay * step)
        return self.er

class QTablePolicy(QBasePolicy):
    def __init__(self,
                 action_space,
                 nS,
                 discount_rate=0.9,
                 learning_rate=0.8,
                 explore_rate_max=0.9,
                 explore_rate_min=0.01,
                 explore_rate_decay=0.005):
        super().__init__(action_space, nS, discount_rate, learning_rate,
                         explore_rate_max, explore_rate_min,
                         explore_rate_decay)

        self.__name__ = 'Q_Table'
        # create a q-table of shape (S X A)
        # representing S X A -> R
        self.nQ = np.append(self.nS, self.nA)
        self.q_table = np.zeros(shape=self.nQ)

    def get_action(self, s):
        # Exploration - Select a random action
        if self.learn_mode and np.random.random() < self.er:
            a = np.random.choice(self.nA)
            if self.debug:
                print('Selecting random action ', a)
        # Exploitation - Select the action with the highest q
        else:
            a = self.q_table[s].argmax()
            if self.debug:
                print('Selecting best action ', a)
        return a

    def update(self, s,a,r,s_,done):
        v_s_ = self.q_table[s_].max()
        e_q_sa = r + self.dr * v_s_
        i = tuple(np.append(s, a))
        self.q_table[i] += self.lr * (e_q_sa - self.q_table[i])
        if self.debug:
            print('Updated Q Table')
            print(self.q_table)

# Experience replay Buffer
import random
random.seed(123)

class Memory():
    def __init__(self, capacity):
        self.cap = capacity
        self.mem = []
        self.ptr = 0

    def append(self, item):
        if len(self.mem) < self.cap:
            self.mem.append(None)
        self.mem[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.cap

    def sample(self, size):
        return random.sample(self.mem, size)

    def __len__(self):
        return len(self.mem)

import torch

class DQNPolicy(QBasePolicy):
    def __init__(self,
                 action_space,
                 nS,
                 discount_rate=0.9,
                 learning_rate=0.8,
                 explore_rate_max=0.9,
                 explore_rate_min=0.01,
                 explore_rate_decay=0.005):
        super().__init__(action_space, nS, discount_rate, learning_rate,
                         explore_rate_max, explore_rate_min,
                         explore_rate_decay)
        self.__name__ = 'DQN'
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        # TODO make batch size passable
        self.batch_size = 128
        memory_capacity = self.batch_size * 10

        # create the empty list to contain game memory
        #        from collections import deque
        #        self.memory = deque(maxlen=memory_len)
        self.memory = Memory(memory_capacity)
        # create a q-nn of shape (S X A)
        # representing S X A -> R
        self.q_nn = self.build_nn()
        self.optimizer = torch.optim.RMSprop(self.q_nn.parameters())
        self.loss_fn = torch.nn.MSELoss()

    def build_nn(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.nS, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, self.nA)).to(self.device)
        return model

    def get_action(self, s):
        # Exploration - Select a random action
        if self.learn_mode and np.random.random() < self.er:
            a = random.randrange(self.nA)
        # Exploitation - Select the action with the highest q
        else:
            with torch.no_grad():
                s = torch.tensor([s],
                                 dtype=torch.float,
                                 device=self.device)
                a = self.q_nn(s).argmax().item()
        return a

    def update(self, s, a, r, s_, done):
        # add the s,a,r,s_ to memory
        if done:
            s_ = None
        self.memory.append([list(s) if isinstance(s,np.ndarray) else [s],
                            [a], [r],
                            list(s_) if isinstance(s_,np.ndarray) else [s_]])
        if len(self.memory) >= self.batch_size:
            sample = self.memory.sample(self.batch_size)
            ss, aa, rr, ss_ = zip(*sample)

            ss = torch.tensor(ss, dtype=torch.float, device=self.device)
            aa = torch.tensor(aa, dtype=torch.long, device=self.device)
            rr = torch.tensor(rr, dtype=torch.float, device=self.device)

            nt_mask = tuple(map(lambda st: st[0] is not None,ss_))
            nt_mask = torch.tensor(nt_mask,dtype=torch.uint8)

            # get all non terminal states
            nt_ss_ = torch.tensor([st for st in ss_ if \
                                   st[0] is not None],
                                  dtype=torch.float,
                                  device=self.device)

            # set all v-values to zero
            v_s_ = torch.zeros(
                len(ss_), 1, device=self.device)
            # set v* for non-terminal states
            v_s_[nt_mask] = self.q_nn(
                nt_ss_).detach().max(1)[0].unsqueeze(1)
            e_q_sa = rr + (self.dr * v_s_)

            q_sa = self.q_nn(ss).gather(1, aa)



            loss = self.loss_fn(q_sa, e_q_sa)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()