
from torch.autograd import Variable
import torch
# from models.initialize_model import initialize_model
import copy
import random

class Client():

    def __init__(self, id, train_loader, test_loader, args, device, clock_t):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.model = initialize_model(args, device)
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        #record local update epoch
        self.epoch = 0
        # record the time
        self.clock = clock_t
        self.CmpTime = round(random.uniform(args.cmin_time, args.cmax_time)*args.time_solt,2)

        self.location = [0,0,0]
        self.l_loss = 0.0

        self.edge_id = -1
        self.actual_times = 0
        self.send_state = 0
        self.received_state = 0


        # correct, total

        self.correct = 0.0
        self.total = 0.0
        self.l_acc = 0.0

    def local_update(self, num_iter, device):
        # print('训练了')
        itered_num = 0
        loss = 0.0
        end = False
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                loss += self.model.optimize_model(input_batch=inputs,
                                                  label_batch=labels)
                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    # print(f"Iterer number {itered_num}")
                    self.epoch += 1
                    self.model.exp_lr_sheduler(epoch=self.epoch)
                    # self.model.print_current_lr()
                    break
            if end: break
            self.epoch += 1
            self.model.exp_lr_sheduler(epoch = self.epoch)
            # self.model.print_current_lr()
        # print(itered_num)
        # print(f'The {self.epoch}')
        loss /= num_iter
        self.generate_actual_times()
        self.send_state = 1
        self.received_state = 0
        self.l_loss = loss
        return loss

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch= inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, edgeserver):
        # self.generate_actual_times()
        edgeserver.receive_from_client(client_id= self.id,
                                        cshared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict()),
                                       a_time=self.actual_times
                                        )
        self.send_state = 0
        return None

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        self.received_state = 1
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.model.update_model(self.receiver_buffer)
        # 改进
        # self.generate_actual_times()

        return None

    def generate_actual_times(self, min_deviation=-1, max_deviation=2):

        deviation = round(random.uniform(min_deviation, max_deviation),2)
        self.actual_times = self.CmpTime + deviation + self.clock


