# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated information back to clients
import copy
from envs.average import average_weights

class SATF():

    def __init__(self, id):
        self.id = id
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.receive_list = 10
        self.send_ist = 10
        self.stor = 100
        self.clock = 0.0
        self.donw_route = {}
        self.t_donw_route = {}
        self.uav_id_registration = []
        self.s_action = 0


        self.avg_acc_v = 0 # The final virtual acc
        self.num_communication = 0 # 云聚合轮次数
        self.accs = 0 #每轮云端聚合的全局精度
        self.losses = 0  # 每轮云端聚合的平均损失
        self.no_receive = []
        # state
        self.refresh_cloudserver_state = 0
        self.send_to_edge_state = 0
        self.send_to_edge_num = 0
        self.sat_aggregate_state = 0
        self.round_state = 0
        self.send_list = []


        self.receive_sate = True


    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.no_receive = []
        self.sample_registration.clear()
        self.round_state = 0
        # self.send_to_edge_state = 0
        self.send_to_edge_num = 0
        self.receive_sate = True
        self.send_list = []

        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def uroute_register(self, edge):
        self.uav_id_registration.append(edge.id)
        # self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        self.receiver_buffer[edge_id] = eshared_state_dict
        return None

    def aggregate(self):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        # print("全局聚合接收到的",len(received_dict))
        sample_num = [snum for snum in self.sample_registration.values()]
        # print("是几个",len(sample_num),sample_num)
        self.shared_state_dict = average_weights(w=received_dict,
                                                 s_num=sample_num)
        self.num_communication += 1
        self.send_to_edge_state = 1
        self.round_state = 1
        # print('全局聚合了',  self.num_communication)
        return None

    def send_to_edge(self, edge, selt):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))

        self.send_list.append(edge.id)
        self.send_to_edge_num += 1
        self.round_state = 0
        if self.send_to_edge_num == selt:
            self.send_to_edge_state = 0
            self.receive_sate = True
            self.send_list = []
        return None

