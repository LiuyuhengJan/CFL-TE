import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import rl_utils
import random
from collections import deque
def mini_batch(transition_dict, batch_size):

    states = torch.tensor(transition_dict['states'], dtype=torch.float)
    # states.requires_grad = False
    actions = torch.tensor(transition_dict['actions']).view(-1, 1)
    rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
    next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
    dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

    # 获取总样本数
    total_samples = states.shape[0]

    for i in range(0, total_samples, batch_size):
        yield {
            'states': states[i:i + batch_size],
            'actions': actions[i:i + batch_size],
            'rewards': rewards[i:i + batch_size],
            'next_states': next_states[i:i + batch_size],
            'dones': dones[i:i + batch_size]
        }
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        assert not torch.isnan(x).any(), "Output logits contain NaN values"
        return F.softmax(x, dim=1)
        # return F.softmax(x, dim=1) + 1e-10  # Add small value for numerical stability


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        # Replay Buffer
        self.replay_buffer = deque(maxlen=500)
        self.batch_size = 20

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def sample_batch(self):
        # batch = random.sample(self.replay_buffer, self.batch_size)
        replay_buffer = list(self.replay_buffer)
        random.shuffle(replay_buffer)
        for i in range(0, len(replay_buffer), self.batch_size):
            batch = replay_buffer[i:i + self.batch_size]
            states, actions, rewards, next_states, dones = zip(*batch)
            yield (
                torch.tensor(states, dtype=torch.float).to(self.device),
                torch.tensor(actions).view(-1, 1).to(self.device),
                torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device),
                torch.tensor(next_states, dtype=torch.float).to(self.device),
                torch.tensor(dones, dtype=torch.float32).view(-1, 1).to(self.device)
            )

    def update(self):
        for states, actions, rewards, next_states, dones in self.sample_batch():
            # 计算 TD 目标
            with torch.no_grad():
                td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            # 计算 TD 误差（Delta）
            td_delta = td_target - self.critic(states)
            # 使用 TD 误差计算优势
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

            # 计算旧的对数概率
            # with torch.no_grad():
            old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

            # PPO 更新循环
            for _ in range(self.epochs):
                # 计算新的对数概率
                log_probs = torch.log(self.actor(states).gather(1, actions))
                # 计算概率比
                ratio = torch.exp(log_probs - old_log_probs)
                # 计算代理损失项
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
                # 计算 actor 损失
                actor_loss = torch.mean(-torch.min(surr1, surr2))
                # 计算 critic 损失
                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

                # 执行反向传播并更新 actor 和 critic 网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        print('final55', 'actor_loss:', actor_loss, 'critic_loss ', critic_loss)
        # print("Initial learning rate: {}".format(self.actor_optimizer.param_groups[0]["lr"]))
        # print("Initial learning rate: {}".format(self.critic_optimizer.param_groups[0]["lr"]))

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

