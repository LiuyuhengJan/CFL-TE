import copy
from average import average_weights
import random
import math
import numpy as np
EARTH_RADIUS = 6371000

class UAV():

    def __init__(self, id):
        
        self.id = id
        self.fl = True # True == 可以聚合 false !=
        self.receiver_buffer = {}
        self.arrive_time = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.location = [0,0,0]
        self.link = None
        self.up_link = None
        self.uplink_state = False
        self.done_state = False
        self.reward = 0
        self.link_state = 0

        self.actual_times = 0
        self.CmpTime = 0

        self.clock = 0

        self.send_state = 0
        self.begin_up_state = 0
        self.arrive_state = False

    def generate_actual_times(self, clock, min_deviation=-5, max_deviation=5):
        self.CmpTime = random.randint(65, 75)
        deviation = random.randint(min_deviation, max_deviation)
        self.actual_times = self.CmpTime + deviation + clock

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



