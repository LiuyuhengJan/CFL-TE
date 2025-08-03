from algorithm import ppo
import torch
from tqdm import tqdm
import numpy as np
from ppoenv import ConFLnet
import os
from obsnorm import RunningMeanStd
import pickle
# from gamebuffer import GameBuffer
num_fea = 21
running_mean_std = RunningMeanStd(num_fea)
# buffer = GameBuffer(max_size=30)

def check_last_dones_true(data_dict):
    for key in data_dict:
        if not data_dict[key]['dones'] or data_dict[key]['dones'][-1] != True:
            return False
    return True
def save_buffer(memory, filename='buffer_20.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(memory, f)


def load_buffer(filename='buffer.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

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
env = ConFLnet(1)
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
            episode_return = 0
            state = env.reset(130000)
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
                    next_state, reward, dones, inf = env.step(action, high_state)

                    all_done = False
                    cc = 0
                    high_state = False


                else:
                    while not all_done:
                        cc += 1
                        all_action = []
                        for i in range(num_e):
                            all_action.append((i,1))
                        next_statel, rewardl, dones = env.step(all_action)
                        next_statel = np.array(next_statel)
                        statel = next_statel
                        episode_return += rewardl[0]
                        all_done = all(dones)
                        if all(dones):
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
        # model_path = os.path.join('hlevel_model_v1.pth')
            #if kais % 100 == 0:
             #   model_path = os.path.join(f'0930/ppo_model_0430_{kais}.pth')
              #  agent.save_model(model_path)


