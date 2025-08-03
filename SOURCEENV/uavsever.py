import copy
from average import average_weights
import random
import math
import numpy as np
EARTH_RADIUS = 6371000

class UAV():

    def __init__(self, id,cids,shared_layers):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.fl = True # True == 可以聚合 false !=
        self.cids = cids
        self.receiver_buffer = {}
        self.arrive_time = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.clock = 0.0
        self.longitude = 10
        self.receive_list = 10
        self.send_ist = 10
        self.stor = 100
        self.location = [0,0,0]


        self.up_route = None
        self.temp_route = None
        self.u_action = 0
        self.state_r = True
        self.u_obs = {}
        self.c_train = True
        self.u_reward = 10
        self.dan_reward = 0

        # state reward
        self.aggregate_state = 0
        self.receive_from_cloudserver_state = 0
        self.send_to_client_state = 0
        self.send_to_cloudserver_state = 0
        # self.client_register_state = 0
        self.refresh_edgeserver_state = 1 # 再看看
        self.up_route_state = 0
        self.selected_clients_state = 0

        self.p_clients = []
        self.edge_loss = 0.0
        self.d_loss = []
        self.edge_sample = 0
        self.correct_all = 0.0
        self.total_all = 0.0

        self.frac = 0.9



    # def FL_ini(self, shared_layers):
    #     self.shared_state_dict = shared_layers.state_dict()

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        self.arrive_time.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        # state
        self.aggregate_state = 0
        self.receive_from_cloudserver_state = 0
        self.send_to_cloudserver_state = 0
        self.send_to_client_state = 1
        self.refresh_edgeserver_state = 0
        self.up_route = None
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)

        self.sample_registration[client.id] = len(client.train_loader.dataset)

        return None

    def receive_from_client(self, client_id, cshared_state_dict, a_time):
        self.receiver_buffer[client_id] = cshared_state_dict
        # self.sample_registration[client.id] = len(client.train_loader.dataset)
        self.arrive_time[client_id] = a_time
        # self.d_loss.append(client.loss)
        # print('aaaaaa', len(self.arrive_time), len(self.cids))
        # print(self.arrive_time)

        if len(self.arrive_time) == len(self.cids):
            self.aggregate_state = 1

        return None

    def aggregate(self):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w = received_dict,
                                                 s_num= sample_num)
        self.aggregate_state = 0
        self.send_to_cloudserver_state = 1
        self.refresh_edgeserver_state = 1

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        self.receive_from_cloudserver_state = 0
        self.send_to_cloudserver_state = 0
        self.send_to_client_state = 0
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict= copy.deepcopy(
                                    self.shared_state_dict))
        self.receive_from_cloudserver_state = 0
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        self.receive_from_cloudserver_state = 1
        self.aggregate_state = 0
        return None

    def selected_clients(self):
        selected_cnum = max(int(len(self.cids) * self.frac), 1)
        self.selected_cids = np.random.choice(self.cids,
                                         selected_cnum,
                                         replace=False,
                                         p=self.p_clients)
        pass

    def ini_loc(self):
        # 随机生成经度（-180到180之间）
        self.location = (random.uniform(20, 50),random.uniform(-9, 9),random.randint(0, 5000))
        # self.UAVPoint()


    def UAVPoint(self):
        radius = EARTH_RADIUS + self.location[2]
        init_pos = [0, 0, 0]
        latitude = math.radians(self.location[0])
        longitude = math.radians(self.location[1])
        init_pos[0] = radius * math.cos(latitude) * math.cos(longitude)
        init_pos[1] = radius * math.cos(latitude) * math.sin(longitude)
        init_pos[2] = radius * math.sin(latitude)
        self.location = init_pos

    # def generate_actual_times(self, min_deviation=-1, max_deviation=2):
    #
    #     deviation = round(random.uniform(min_deviation, max_deviation),2)
    #     self.actual_times = self.CmpTime + deviation + self.clock



