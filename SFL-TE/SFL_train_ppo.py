from config import get_config
from obsnorm import RunningMeanStd
def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=parser.parse_args().num_edges, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args

parser = get_config().parse_known_args()[0]


from envs.env_runner_trainensac import EnvRuuner


from algorithm import ppo
import torch
# import gym
from tqdm import tqdm
import numpy as np
def save_model(agent, save_path="sac_model.pth"):
    torch.save({
        # 'episode': episode,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
        'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_1_state_dict': agent.critic_optimizer_1.state_dict(),
        'critic_optimizer_2_state_dict': agent.critic_optimizer_2.state_dict(),
        'log_alpha': agent.log_alpha,
        'alpha_optimizer_state_dict': agent.alpha_optimizer.state_dict()
    }, save_path)
    # print(f"Model saved after episode {episode} at {save_path}")

import os
def check_last_dones_true(data_dict):
    for key in data_dict:
        if not data_dict[key]['dones'] or data_dict[key]['dones'][-1] != True:
            return False
    return True
num_e = 5
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 10000
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
print('device', device)
torch.manual_seed(66)
num_fea = 21
running_mean_std = RunningMeanStd(num_fea)

num_e = 5
actor_lr = 1e-4
critic_lr = 1e-4
num_episodes = 1000
hidden_dim = 256
gamma = 0.99
lmbda = 0.95
epochs = 2
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
print('device',device)
# env_name = 'CartPole-v0'
env = EnvRuuner(parser)
# env.seed(0)
torch.manual_seed(66)
state_dim = 22
action_dim = 26
agent = ppo.PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

return_list = []
kais = 0
allstep = 0
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            state = env.reset()
            state = np.array(state)
            features = state[ :-1]
            binary_vars = state[ -1:]
            running_mean_std.update(features)
            normalized_features = running_mean_std.normalize(features)
            state = np.hstack((normalized_features, binary_vars))

            high_state = True
            fl_done = False
            while not fl_done:
                if high_state:
                    # action, hinstate = agent.take_action(state, h_0)
                    action = agent.take_action(state)
                    next_state, reward, dones = env.step(action,False,True)

                    all_done = False
                    cc = 0
                    high_state = False


                else:
                    # while not all_done:
                    cc += 1
                    all_action = []
                    for i in range(num_e):
                        all_action.append((i,1))
                    next_statel, rewardl, dones = env.step(all_action)
                    next_statel = np.array(next_statel)
                    statel = next_statel
                    # all_done = all(dones)
                    if dones:
                        high_state = True
                        nextstate, flreward, fl_done = env.round_reset()
                        # print('round done', flreward)
                        nextstate = np.array(nextstate)
                        features = nextstate[:-1]
                        binary_vars = nextstate[-1:]
                        running_mean_std.update(features)
                        normalized_features = running_mean_std.normalize(features)
                        nextstate = np.hstack((normalized_features, binary_vars))
                        agent.store_transition((state, action, flreward, nextstate, fl_done))
                        state = nextstate
                        return_list.append(flreward)
                        allstep += 1

                        if allstep % 320 == 0:
                            agent.update()
                            # save_buffer(agent.replay_buffer, filename=f'0930/buffer_0929_{allstep}_acton26.pkl')
                            agent.replay_buffer.clear()

            kais += 1

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

