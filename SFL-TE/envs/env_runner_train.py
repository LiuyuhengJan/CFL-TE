
import math
import networkx as nx
from envs.Ground_A import Client
# from constellation import Constellation
from datasets.get_data import get_dataloaders, show_distribution
import numpy as np
from envs.uavsever import UAV
import copy
import time
from models.mnist_cnn import mnist_lenet
from models.cifar_cnn_3conv_layer import cifar_cnn_3conv
from models.cifar_resnet import ResNet18
from models.mnist_logistic import LogisticRegression
import torch
from envs.constellation import Constellation
import random
import Fencoding
from envs import getaction
# import envs.lowagent as lowagent

# the mean radius of the earth in meters according to wikipedia
EARTH_RADIUS = 6371000

# earth's z axis (eg a vector in the positive z direction)
EARTH_ROTATION_AXIS = [0, 0, 1]

# number of seconds per earth rotation (day)
SECONDS_PER_DAY = 86400

# according to wikipedia
STD_GRAVITATIONAL_PARAMATER_EARTH = 3.986004418e14

# how big to initialize the ground point array...
NUM_GROUND_POINTS = 0

max_isl_distance=4620303
max_stg_distance=741292
uav_loc = []
with open('./envs/city_data.txt', 'r') as file:
    for line in file:
        my_line = []
        for word in line.split():
            my_line.append(word)
        uav_loc.append(my_line)
def get_client_class(args, clients):
    client_class = []
    client_class_dis = [[],[],[],[],[],[],[],[],[],[]]
    for client in clients:
        train_loader = client.train_loader
        distribution = show_distribution(train_loader, args)
        label = np.argmax(distribution)
        client_class.append(label)
        client_class_dis[label].append(client.id)
    print(client_class_dis)
    return client_class_dis


def initialize_edges_iid(num_edges, clients, args, client_class_dis):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user,
    but the distribution among edges is iid,
    10 clients per edge, each edge have 10 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:
    """
    #only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        for label in range(10):
        #     0-9 labels in total
            assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace = False)
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
        edges.append(UAV(id = eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                        for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
    #And the last one, eid == num_edges -1
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(UAV(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                    for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients
def initialize_global_nn(args):
    if args.dataset == 'mnist':
        if args.model == 'lenet':
            global_nn = mnist_lenet(input_channels=1, output_channels=10)
        elif args.model == 'logistic':
            global_nn = LogisticRegression(input_dim=1, output_dim=10)
        else: raise ValueError(f"Model{args.model} not implemented for mnist")
    elif args.dataset == 'cifar10':
        if args.model == 'cnn_complex':
            global_nn = cifar_cnn_3conv(input_channels=3, output_channels=10)
        elif args.model == 'resnet18':
            global_nn = ResNet18()
        else: raise ValueError(f"Model{args.model} not implemented for cifar")
    else: raise ValueError(f"Dataset {args.dataset} Not implemented")
    return global_nn


def initialize_edges_niid(num_edges, clients, args, client_class_dis):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 5 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:
    """
    #only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    label_ranges = [[0,1,2,3,4],[1,2,3,4,5],[5,6,7,8,9],[6,7,8,9,0]]
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        label_range = label_ranges[eid]
        for i in range(2):
            for label in label_range:
                #     5 labels in total
                if len(client_class_dis[label]) > 0:
                    assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace=False)
                    client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
                else:
                    label_backup = 2
                    assigned_client_idx = np.random.choice(client_class_dis[label_backup],1, replace=False)
                    client_class_dis[label_backup] = list(set(client_class_dis[label_backup]) - set(assigned_client_idx))
                for idx in assigned_client_idx:
                    assigned_clients_idxes.append(idx)
        edges.append(UAV(id = eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                        for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
    #And the last one, eid == num_edges -1
    #Find the last available labels
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(UAV(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                    for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients


def all_clients_test(server, clients, cids, device):
    [server.send_to_client(clients[cid]) for cid in cids]
    for cid in cids:
        server.send_to_client(clients[cid])
        # The following sentence!
        clients[cid].sync_with_edgeserver()
    correct_edge = 0.0
    total_edge = 0.0
    for cid in cids:
        correct, total = clients[cid].test_model(device)
        correct_edge += correct
        total_edge += total
    return correct_edge, total_edge


def fast_all_clients_test(v_test_loader, global_nn, device):
    correct_all = 0.0
    total_all = 0.0
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = global_nn(inputs)
            _, predicts = torch.max(outputs, 1)
            total_all += labels.size(0)
            correct_all += (predicts == labels).sum().item()
    return correct_all, total_all


def addGroundPoint(latitude, longitude, altitude=100.0):

    # must convert the lat/long/alt to cartesian coordinates
    radius = EARTH_RADIUS + altitude
    init_pos = [0, 0, 0]
    latitude = math.radians(latitude)
    longitude = math.radians(longitude)
    init_pos[0] = radius * math.cos(latitude) * math.cos(longitude)
    init_pos[1] = radius * math.cos(latitude) * math.sin(longitude)
    init_pos[2] = radius * math.sin(latitude)



# reset
def generate_device_times(devices, min_time, max_time):


    for device in devices:
        device.CmpTime = random.uniform(min_time, max_time)

min_communications_altitude = 100000
min_sat_elevation = 40




class EnvRuuner():
    def __init__(self, Config):
        # super(EnvRuuner, self).__init__(Config)
        self.args = Config

        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)
            cuda_to_use = torch.device(f'cuda:{self.args.gpu}')
        self.device = cuda_to_use if torch.cuda.is_available() else "cpu"


    def reset(self, time = -1):
        if time == -1:
            self.ini_time = np.random.randint(0,20)*10000 #np.random.randint(0,3000)
            print('全局开始',self.ini_time)
        else:
            self.ini_time = time
        self.gl_times = {}
        self.clock_time = self.args.init_time
        self.Con = Constellation(self.args)
        self.max_isl_distance = self.Con.calculateMaxISLDistance(min_communications_altitude)  # 优化
        # print('max_isl_distance', max_isl_distance)
        self.max_stg_distance = self.Con.calculateMaxSpaceToGndDistance(min_sat_elevation)  # 优化
        # print('max_stg_distance', max_stg_distance)
        self.local_times = self.args.num_local_update
        device = self.device
        print('初始化',self.clock_time)
        print("device", device)
        print('星座',self.Con.nodes_per_plane,self.Con.number_of_planes)

        train_loaders, test_loaders, v_train_loader, self.v_test_loader = get_dataloaders(self.args)
        self.clients = []
        for i in range(self.args.num_clients):
            # 初始化客户端
            self.clients.append(Client(id=i,
                                  train_loader=train_loaders[i],
                                  test_loader=test_loaders[i],
                                  args=self.args,
                                  device=device,
                                    clock_t= self.clock_time,)
                           )

        initilize_parameters = list(self.clients[0].model.shared_layers.parameters())
        nc = len(initilize_parameters)
        for client in self.clients:
            user_parameters = list(client.model.shared_layers.parameters())
            for i in range(nc):
                user_parameters[i].data[:] = initilize_parameters[i].data[:]

        # 初始化边缘
        self.uav_names = []
        self.edges = []
        self.ok_num = 0
        self.round_ok_num = 0

        cids = np.arange(self.args.num_clients)
        self.clients_per_edge = int(self.args.num_clients / self.args.num_edges)
        self.p_clients = [0.0] * self.args.num_edges

        if self.args.iid == -2:
            if self.args.edgeiid == 1:
                print('self.args.edgeiid == 1')
                client_class_dis = get_client_class(self.args, self.clients)
                self.edges, self.p_clients = initialize_edges_iid(num_edges=self.args.num_edges,
                                                        clients=self.clients,
                                                        args=self.args,
                                                        client_class_dis=client_class_dis)
            elif self.args.edgeiid == 0:
                client_class_dis = get_client_class(self.args, self.clients)
                self.edges, self.p_clients = initialize_edges_niid(num_edges=self.args.num_edges,
                                                         clients=self.clients,
                                                         args=self.args,
                                                         client_class_dis=client_class_dis)
        else:
            # This is randomly assign the clients to edges
            print('This is randomly assign the clients to edges')
            for i in range(self.args.num_edges):
                # uav_names.append(f"UAV{i}")
                # Randomly select clients and assign them
                selected_cids = np.random.choice(cids, self.clients_per_edge, replace=False)
                cids = list(set(cids) - set(selected_cids))
                self.edges.append(UAV(id=i,
                                 cids=selected_cids,
                                 shared_layers=copy.deepcopy(self.clients[0].model.shared_layers)))
                [self.edges[i].client_register(self.clients[cid]) for cid in selected_cids]
                for cid in selected_cids:
                    self.clients[cid].edge_id = self.edges[i].id
                self.edges[i].all_trainsample_num = sum(self.edges[i].sample_registration.values())
                self.p_clients[i] = [sample / float(self.edges[i].all_trainsample_num) for sample in
                                list(self.edges[i].sample_registration.values())]
                self.edges[i].p_clients = self.p_clients[i]
                self.edges[i].refresh_edgeserver()
                self.gl_times[i] = self.args.init_time
                # edges[i].ini_loc()
                # model.addGroundPoint(edges[i].location[0],edges[i].location[1])

        for i in range(self.args.num_edges):
            self.uav_names.append(f"UAV{i}")
            self.edges[i].clock = self.clock_time
            self.edges[i].location = [float(uav_loc[i+1][1]),float(uav_loc[i+1][2]),1.0]
            self.Con.addGroundPoint(self.edges[i].location[0], self.edges[i].location[1])

        self.Con.generateNetworkGraph(self.uav_names)

        self.Con.calculateIdealLinks(max_isl_distance, max_stg_distance)
        self.global_nn = initialize_global_nn(self.args)
        for _ in self.Con.FLs:
            _.clock = self.clock_time
        if self.args.cuda:
            self.global_nn = self.global_nn.cuda(device)
        self.selected_nodes = []
        self.step_edges = []
        self.Con.FLs[0].refresh_cloudserver()
        self.round_dealine = 500
        self.round_time = 0
        self.global_times = 0
        self.fl_done = False
        self.edges_obs = self.get_obs_a()
        # self.take_action = lowagent.get_agent()
        from algorithm import test_a
        from envs import ppoconfig

        all_args = ppoconfig.get_parser().parse_known_args()[0]

        self.take_action = test_a.PPO(all_args)
        self.round_done = False

        # self.take_action.load_model('ppo_model_random.pth')



    def step(self, action, done = False):
        # print('clock_time',self.clock_time)
        # self.round_done = done

        shibai = 0
        binary_list = getaction.get_combination_by_index(action)
        # binary_list = getaction.get_combination_by_index(15)
        binary_a = np.array(binary_list)
        self.selected_nodes = np.where(binary_a == 1)[0]
        print('ssss', self.selected_nodes)

        while not done:
            # 卫星部分
            if len(self.Con.FLs[0].receiver_buffer) + len(self.Con.FLs[0].no_receive) == len(self.selected_nodes):
                shibai = shibai + 1
                print('接收了',len(self.Con.FLs[0].receiver_buffer))
                print('发送失败了',len(self.Con.FLs[0].no_receive))
                temp_acc = 0.0
                client_test_num = 0
                for ccc in self.clients:
                    if ccc.edge_id in self.selected_nodes:
                        ccc.test_state = 1
                        client_test_num += 1
                        temp_acc += ccc.c_acc
                all_acc = temp_acc / client_test_num
                print('all acc is', all_acc)
                print('时间',self.clock_time)
                reward = 0
                FL_done = False
                if all_acc > 0.9:
                    FL_done = True
                    reward = 1
                    print('全部完成')


                if (len(self.Con.FLs[0].no_receive) == len(self.selected_nodes)):
                    self.Con.FLs[0].receive_sate = False
                    print('没进行全局聚合')
                    self.Con.FLs[0].no_receive = []
                else:
                    self.Con.FLs[0].aggregate()
                    self.Con.FLs[0].no_receive = []
                    self.Con.FLs[0].refresh_cloudserver()
                    # h_act = random.randint(0, 15)
                    # binary_list = getaction.get_combination_by_index(h_act)
                    # binary_a = np.array(binary_list)
                    # self.selected_nodes = np.where(binary_a == 1)[0]
                    # print('ssss', self.selected_nodes)
                obs = self.get_sat_obs_nlink()
                return obs, reward, FL_done

            # 边缘
            for i, edge in enumerate(self.edges):

                if i in self.selected_nodes:

                    edge.temp_route = self.link_here(i, self.Con.FLs[0].id)
                    # print('edge.temp_route',edge.temp_route)
                    if edge.temp_route != None:
                        edge.up_route_state = 1
                        # print('edge.temp_route',edge.id,edge.temp_route)
                        # print('edge.up_route_state ', i,edge.up_route_state)
                    else:
                        edge.up_route_state = 0

                    if edge.up_route != None and edge.up_route != 'ok' and edge.up_route != 'wu':
                        edge.up_route = self.can_transmit(edge)  # tcan_transmi
                        # print(f'self.edges {i}.up_route', self.edges[i].up_route)

                    if edge.up_route_state == 1 and self.Con.FLs[0].send_to_edge_num <5  and self.Con.FLs[0].receive_sate==True:
                        if edge.receive_from_cloudserver_state == 0 and self.Con.FLs[0].send_to_edge_state == 1:
                            if edge.id not in self.Con.FLs[0].send_list:
                                self.Con.FLs[0].send_to_edge(edge, len(self.selected_nodes))

                    if self.edges[i].up_route == 'ok':
                        # print('self.edges[i].id',self.edges[i].id)
                        if self.edges[i].id not in self.Con.FLs[0].id_registration:
                            self.Con.FLs[0].edge_register(edge)
                            edge.send_to_cloudserver(self.Con.FLs[0])
                            self.edges[i].up_route = None
                    elif self.edges[i].up_route == 'wu' and self.round_time < self.round_dealine:
                        if edge.id not in self.Con.FLs[0].no_receive:
                            self.Con.FLs[0].no_receive.append(edge.id)
                            edge.receive_from_cloudserver_state = 0
                            self.edges[i].up_route = None

                    # 发给sat
                    if edge.up_route_state == 1 and edge.aggregate_times == 4 and edge.up_route == None :
                        # 加在这加载模型判断
                        if self.take_action.take_action(self.edges_obs[edge.id]) == 1:
                            edge.up_route = edge.temp_route
                            edge.aggregate_times = 5
                        else:
                            print(edge.id, '等待')

                    if self.Con.FLs[0].receive_sate == False:
                        if edge.up_route_state == 1 and edge.id not in self.Con.FLs[0].id_registration and edge.up_route == None:
                            edge.up_route = edge.temp_route
                        # edge.aggregate_times == 4
                        # edge.up_route == None

                    # 发给用户
                    if edge.send_to_client_state == 1 and edge.aggregate_times < 4 and self.Con.FLs[0].receive_sate == True:
                        # edge.aggregate()
                        edge.refresh_edgeserver()

                        for cc in edge.cids:
                            edge.send_to_client(self.clients[cc])
                            self.clients[cc].sync_with_edgeserver()
                    # 边缘聚合
                    if edge.aggregate_state == 1:
                        edge.aggregate()


            # 用户
            for i, client in enumerate(self.clients):
                if client.edge_id in self.selected_nodes:
                # print('用户')
                # print(client.id, client.received_state)
                    if client.send_state == 1 and client.actual_times <= self.clock_time and client.id not in self.edges[
                        client.edge_id].receiver_buffer.keys():
                        client.send_to_edgeserver(self.edges[client.edge_id])
                        # print('send', client.id, self.clock_time)
                        self.edges[client.edge_id].d_loss.append(client.l_loss)
                        self.edges[client.edge_id].client_register(client)

                    if client.received_state == 1 and self.Con.FLs[0].receive_sate==True:
                        # print('update', client.id, self.clock_time)
                        if client.test_state == 1:
                            client.test_model(self.device)
                        # 可控制
                        # print(client.id,client.received_state)
                        client.local_update(self.args.num_local_update, self.device)
                        # print(client.id, client.received_state)

                        # client.test_model(self.device)




            self.update_clock_time()
            # 加观测
            self.edges_obs = self.get_obs_a()



    #要换
    def get_obs_a(self):
        obs = []
        cc = self.Con.link_array[self.Con.number_of_isl_links:self.Con.total_links]
        for i, ss in enumerate(self.Con.groundpoints_array):
            tb_obs = []
            tb_obs.append(self.clock_time)
            tb_obs.append('*')
            tb_obs.append(ss[1])
            tb_obs.append(ss[2])
            tb_obs.append(ss[3])
            tb_obs.append('#')
            tb_obs.append(self.Con.FLs[0].id)
            tb_obs.append(self.Con.satellites_array[self.Con.FLs[0].id]['x'])
            tb_obs.append(self.Con.satellites_array[self.Con.FLs[0].id]['y'])
            tb_obs.append(self.Con.satellites_array[self.Con.FLs[0].id]['z'])
            tb_obs.append('&')
            # self.edges[i].link = self.link_here(i, self.Con.FLs[0].id)
            for inll in cc:
                if -(i + 1) == inll[0]:
                    tb_obs.append(inll[1])
                    tb_obs.append(self.Con.satellites_array[inll[1]]['x'])
                    tb_obs.append(self.Con.satellites_array[inll[1]]['y'])
                    tb_obs.append(self.Con.satellites_array[inll[1]]['z'])
                    tb_obs.append(inll[2])

            all_obs = tb_obs

            allb = []
            for ss in all_obs:
                # print(ss)
                kk = Fencoding.split_number(ss)
                # print(kk)
                for k in kk:
                    allb.append(k)
            # print(len(allb))
            while len(allb) < 51:
                allb.append(2010)
            if len(allb) > 52:
                print('还是小')
            int_list = [int(num) for num in allb]

            obs.append(int_list)

        return obs

    def get_sat_obs_nlink(self):
        obs = []
        obs.append(self.clock_time)
        obs.append('#')
        obs.append(self.Con.satellites_array[self.Con.FLs[0].id]['ID'])
        obs.append(self.Con.satellites_array[self.Con.FLs[0].id]['plane_number'])
        obs.append(self.Con.satellites_array[self.Con.FLs[0].id]['offset_number'])
        obs.append(self.Con.satellites_array[self.Con.FLs[0].id]['time_offset'])
        obs.append(self.Con.satellites_array[self.Con.FLs[0].id]['x'])
        obs.append(self.Con.satellites_array[self.Con.FLs[0].id]['y'])
        obs.append(self.Con.satellites_array[self.Con.FLs[0].id]['z'])
        obs.append(self.ok_num)
        obs.append(self.round_ok_num)

        if self.fl_done:
            obs.append('!')
        else:
            obs.append('^')

        for i, ss in enumerate(self.Con.groundpoints_array):
            obs.append("&")
            if i in self.selected_nodes:
                obs.append('!')
            else:
                obs.append('^')
            obs.append(ss[1])
            obs.append(ss[2])
            obs.append(ss[3])
            if self.edges[i].arrive_state == True:
                obs.append('!')
            else:
                obs.append('^')

        allb = []
        for ss in obs:
            # print(ss)
            kk = Fencoding.split_number(ss)
            # print(kk)
            for k in kk:
                allb.append(k)
        # print(len(allb))
        while len(allb) < 93:
            allb.append(2010)
        if len(allb) > 93:
            print('还是小1')
        int_list = [int(num) for num in allb]
        # obs.append(tb_obs)

        obs = int_list
        return obs

    def update_clock_time(self):
        self.clock_time += self.args.time_solt

        if self.clock_time % 1000 == 0:
            print('.clock_time',self.clock_time)
            if self.Con.FLs[0].receive_sate == False:
                for edge in self.edges:
                    print(edge.up_route, edge.temp_route, edge.up_route_state)

        for _ in self.clients:
            _.clock = self.clock_time
            # print('client', _.clock)
        for _ in self.edges:
            _.clock = self.clock_time
            # print('edge', _.clock)
        for _ in self.Con.FLs:
            _.clock = self.clock_time
            # print('FL', _.clock)
        self.Con_update()
        pass

    def Con_update(self):
        # print('self.clock_time',self.clock_time)
        self.Con.setConstillationTime(self.clock_time)
        self.Con.generateNetworkGraph(self.uav_names)
        self.Con.calculateIdealLinks(
            self.max_isl_distance,
            self.max_stg_distance)
        # print("////")



    def updatelink(self, node_1, node_2):
        # update links / network design / sat positions

        path_links = {}

        for s in node_1:
            # try:

            path_links[s] = self.link_here(s, node_2)

        return path_links

    def link_here(self, n1, n2):
        try:
            path = nx.shortest_path(
                self.Con.G,
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
        path = edge.up_route
        print('path', edge.id, path)
        # print(edge.link)
        edge_data = self.Con.G.get_edge_data(path[0][0], path[0][1])  # 有时候会不存在边
        # print(edge_data)
        if edge_data != None:

            # if random.random() < self.Con.satellites_array[int(path[0][1])]['pro_capacity']:
            if 0.7 < self.Con.satellites_array[int(path[0][1])]['pro_capacity']:
                # del path[0]
                del path[0]
                # print('错了',pa)

                if path == []:
                    path = 'ok'
                    print(edge.id,'pa', path)
                return path
            else:
                print(edge.id,'传输断开wu')

                path = 'wu'
                return path
        else:
            print(edge.id,'链路没了wu')
            path = 'wu'
            return path





