import copy

from options import args_parser
# from tensorboardX import SummaryWriter
# import torch
from Ground_A import Client
from constellation import Constellation
import networkx as nx
import random
from edgedevice import UAV
from obsnorm import RunningMeanStd
import getaction
from gym import spaces
import numpy as np
class ConFLnet():
    def __init__(self, Config):
        self.args = Config
        self.observation_space = 1160
        self.sat_observation_space = 1171

        # self.signal_obs_dim = 4
        self.agent_num = 5
        self.signal_action_dim = 2
        self.action_space = spaces.Discrete(self.signal_action_dim)
        self.high_action_space = spaces.MultiBinary(self.agent_num)
    def reset(self, time = -1):
        # 时间
        if time == -1:
            self.ini_time = np.random.randint(0,20)*10000
            print('全局开始',self.ini_time)
        else:
            self.ini_time = time

        self.model = Constellation(
                planes=6,
                nodes_per_plane=11,
                inclination=65.0,
                semi_major_axis= 6871000.0,
                minCommunicationsAltitude=100000,
                minSatElevation=30, linkingMethod = 30)


        groundPtsFile='city_data.txt'
        data = []
        city_names = []
        with open(groundPtsFile, 'r') as f:
                for line in f:
                    my_line = []
                    for word in line.split():
                        my_line.append(word)
                    data.append(my_line)
        self.all_agent = []
        for i in range(1, len(data)):
                city_names.append(data[i][0])
                self.model.addGroundPoint(float(data[i][1]), float(data[i][2]))
        self.uav_names = city_names

        for i in range(len(self.uav_names)):
            self.all_agent.append(UAV(i))
            self.all_agent[i].location = [self.model.groundpoints_array[i][1], self.model.groundpoints_array[i][2],
                                          self.model.groundpoints_array[i][3]]
            # self.all_agent[i].up_time = self.ini_time + round(random.uniform(30,50),2)
            # self.all_agent[i].clock = self.ini_time

        # max_isl_range
        # 4620303
        # max_stg_range
        # 909424
        min_communications_altitude = 100000
        min_sat_elevation = 30
        self.max_isl_distance = self.model.calculateMaxISLDistance(min_communications_altitude)

        self.max_stg_distance = self.model.calculateMaxSpaceToGndDistance(min_sat_elevation)

        # obs =np.random.randn(self.observation_space)
        self.link = []

        self.clock_time = self.ini_time
        self.location_id = random.randint(1, len(self.uav_names)-1)
        self.Con_update()
        # obs = self.get_sat_obs()
        self.selected_nodes = []
        self.ok_num = 0
        self.fl_done = False
        self.round_ok_num = 0
        self.fl_train_time = 0
        self.ini_up_time = self.ini_time
        obs = self.get_sat_obs_nlink()

        # print('开始',self.clock_time)

        self.round_reward = 0
        return obs

    def round_reset(self):
        avgtime = 10000 / 40
        reward_t = self.round_ok_num * avgtime - self.round_time
        reward = self.round_ok_num * 5 + reward_t / avgtime

        if self.ok_num > 40:
            self.fl_train_time = self.clock_time - self.ini_time
            self.fl_done = True
            print(self.ini_time, '一次结束', self.fl_train_time, self.ok_num, self.selected_nodes)
            if self.fl_train_time > 10000:
                reward = -((self.fl_train_time / avgtime) ** 0.5)*20 + reward
            else:
                reward = (10000 - self.fl_train_time)/10 + 100 + reward
        else:
            self.round_reward += reward

        reward = round(reward, 2)
        print('ok_num:', self.ok_num, self.round_ok_num, self.selected_nodes, reward,
              self.round_time)

        # self.round_reward = 0
        self.ini_up_time = self.clock_time
        self.Con_update()
        obs = self.get_sat_obs_nlink()
        self.round_ok_num = 0
        for edge in self.all_agent:
            edge.reward = 0.0
            edge.uplink_state = False
            edge.done_state = False
            edge.link_state = 0
            edge.up_link = None
            edge.link = None
            edge.begin_up_state = 0
            edge.arrive_state = False
        return obs, reward, self.fl_done

    def step(self, actions, high_level=False):
        self.round_time = self.clock_time - self.ini_up_time
        if high_level:
            binary_list = getaction.get_combination_by_index(actions)
            binary_a = np.array(binary_list)
            self.selected_nodes = np.where(binary_a == 1)[0]
            # print('ssss', self.selected_nodes)
            obs = []
            reward = 0
            for edge in self.all_agent:
                reward += edge.reward


            self.Con_update()
            obs = self.get_sat_obs_nlink()
            inf = False
            return obs, reward, self.fl_done, inf
        else:
            for i, action in actions:
                if i not in self.selected_nodes:
                    self.all_agent[i].done_state = True
                else:
                    if action == 0:
                        self.all_agent[i].reward = 0.0
                        # self.all_agent[i].done_state = False
                    elif action == 1:
                        if self.all_agent[i].link != None and self.all_agent[i].uplink_state == False:
                            if self.all_agent[i].begin_up_state == 1 and self.all_agent[i].actual_times< self.clock_time:
                                self.all_agent[i].uplink_state = True
                                self.all_agent[i].up_link = self.all_agent[i].link
                        else:
                            self.all_agent[i].reward = 0.0
                            # self.all_agent[i].done_state = False

            for edge in self.all_agent:
                if edge.uplink_state == True and edge.done_state != True:
                    edge.up_link = self.can_transmit(edge)
                    if edge.up_link == 'ok':
                        edge.reward = 10
                        self.ok_num +=1
                        self.round_ok_num+=1
                        edge.arrive_state = True
                        # print('self.ok_num',self.ok_num)
                        edge.done_state = True
                    elif edge.up_link == 'wu':
                        edge.reward = -1
                        edge.done_state = True

                # if self.round_time>500 and edge.done_state != True:
                #     if edge.link_state == 1:
                #         self.round_reward += -10
                #         edge.done_state = True
                #         edge.reward = 0.0
                #     else:
                #         edge.done_state = True
                #         edge.reward = 0.0

                if edge.link != None:
                    if edge.begin_up_state == 0 and edge.id in self.selected_nodes:
                        edge.generate_actual_times(self.clock_time)
                        # print('发送给',edge.id,edge.actual_times)
                        edge.begin_up_state = 1


            self.Con_update()
            dones = []
            reward = []
            for edge in self.all_agent:
                dones.append(edge.done_state)
                reward.append(edge.reward)
            obs = []
            return obs, reward, dones


    def get_sat_obs_nlink(self):
        obs = []
        obs.append(self.clock_time)
        obs.append(self.model.satellites_array[self.model.FLs[0].id]['x'])
        obs.append(self.model.satellites_array[self.model.FLs[0].id]['y'])
        obs.append(self.model.satellites_array[self.model.FLs[0].id]['z'])
        obs.append(self.ok_num)
        obs.append(self.round_ok_num)
        for i, ss in enumerate(self.model.groundpoints_array):
            obs.append(ss[1])
            obs.append(ss[2])
            obs.append(ss[3])

        if self.fl_done:
            obs.append(1)
        else:
            obs.append(0)
        return obs



    def Con_update(self):
        # print('self.clock_time',self.clock_time)
        self.clock_time += 1
        self.model.setConstillationTime(self.clock_time)
        self.model.generateNetworkGraph(self.uav_names)
        self.model.calculateIdealLinks(
            self.max_isl_distance,
            self.max_stg_distance)
        for i, edge in enumerate(self.all_agent):
            edge.link = self.link_here(i, self.model.FLs[0].id)
            if edge.link != None:
                edge.link_state = 1

    def updatelink(self, node_1, node_2):
        # update links / network design / sat positions

        # path_links = {}

        # for s in node_1:
            # try:

        path_links = self.link_here(node_1, node_2)

        return path_links

    def link_here(self, n1, n2):
        try:
            path = nx.shortest_path(
                self.model.G,
                source=str(-(n1 + 1)),
                target=str(n2),
                weight='distance')
            path_links_t = []
            for i in range(len(path) - 1):
                path_links_t.append([path[i], path[i + 1]])


        except nx.exception.NetworkXNoPath:
            path_links_t = None
        return path_links_t

    def can_transmit(self, edge):
        path = edge.up_link
        edge_data = self.model.G.get_edge_data(path[0][0], path[0][1])  # 有时候会不存在边
        if edge_data != None:
            if 0.7 < self.model.satellites_array[int(path[0][1])]['pro_capacity']:
            # if random.random() < self.model.satellites_array[int(path[0][1])]['pro_capacity']:
                # del path[0]
                del path[0]

                if path == []:
                    path = 'ok'
                    # print(edge.id, path)
                return path
            else:
                # print(edge.id, '传输断开')
                path = 'wu'
                return path
        else:
            # print(edge.id, '链路没了')
            path = 'wu'
            return path



