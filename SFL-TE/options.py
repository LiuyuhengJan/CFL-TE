import argparse
import os

import torch

def args_parser():
    parser = argparse.ArgumentParser()
    #dataset and model
    parser.add_argument(
        '--dataset',
        type = str,
        default = 'mnist',
        help = 'name of the dataset: mnist, cifar10'
    )
    parser.add_argument(
        '--model',
        type = str,
        default = 'lenet',
        help='name of model. mnist: logistic, lenet; cifar10: resnet18, cnn_complex'
    )
    parser.add_argument(
        '--input_channels',
        type = int,
        default = 1,
        help = 'input channels. mnist:1, cifar10 :3'
    )
    parser.add_argument(
        '--output_channels',
        type = int,
        default = 10,
        help = 'output channels'
    )
    #nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 20,
        help = 'batch size when trained on client'
    )
    # -------------云聚合轮次、边缘聚合轮次、本地更新轮次
    parser.add_argument(
        '--num_communication',
        type = int,
        default=300,
        help = 'number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type = int,
        default=4,
        help = 'number of edge aggregation (K_2)'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=15,
        help='number of local update (K_1)'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.01, # 0.001
        help = 'learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type = float,
        default= '0.995',
        help = 'lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type = int,
        default=1,
        help= 'lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type = float,
        default = 0.9,
        help = 'SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type = float,
        default = 0.0001, # 0
        help= 'The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type = int,
        default = 0,
        help = 'verbose for print progress bar'
    )
    #setting for federeated learning
    parser.add_argument(
        '--iid',
        type = int,
        default = 1,
        help = 'distribution of the data, 1,0, -2(one-class)'
    )
    parser.add_argument(
        '--edgeiid',
        type=int,
        default=1,
        help='distribution of the data under edges, 1 (edgeiid),0 (edgeniid) (used only when iid = -2)'
    )
    parser.add_argument(
        '--frac',
        type = float,
        default = 1,
        help = 'fraction of participated clients'
    )
    # -------------客户端数、边缘服务器数、客户端训练样本量
    parser.add_argument(
        '--num_clients',
        type = int,
        default = 50,
        help = 'number of all available clients'
    )
    parser.add_argument(
        '--num_edges',
        type = int,
        default= 5,
        help= 'number of edges'
    )
    parser.add_argument(
        '--num_sample_per_client',
        default= 1200,
        type=int,
        help='>=0: number of samples per client， -1: all samples'
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 1,
        help = 'random seed (defaul: 1)'
    )

    # editer: Sensorjang 20230925
    dataset_root = os.path.join(os.getcwd(), 'train_data')
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    parser.add_argument(
        '--dataset_root',
        type = str,
        default = dataset_root,
        help = 'dataset root folder'
    )
    parser.add_argument(
        '--show_dis',
        type= int,
        default= False,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--classes_per_client',
        type=int,
        default = 2,
        help='under artificial non-iid distribution, the classes per client'
    )
    parser.add_argument(
        '--gpu',
        type = int,
        default=0,
        help = 'GPU to be selected, 0, 1, 2, 3'
    )

    parser.add_argument(
        '--mtl_model',
        default=0,
        type = int
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )

    # editer: Sensorjang 20230925
    parser.add_argument(
        '--test_on_all_samples',
        type = int,
        default = 0,
        help = '1 means test on all samples, 0 means test samples will be split averagely to each client'
    )
    # Env
    parser.add_argument(
        '--planes',
        default=3,
        type=int
    )

    parser.add_argument(
        '--nodes_per_plane',
        default=11,
        type=int
    )

    parser.add_argument(
        '--inclination',
        default=65,
        type=int
    )

    parser.add_argument(
        '--semi_major_axis',
        default=6871000.0,
        type=float
    )

    parser.add_argument(
        '--minCommunicationsAltitude',
        default=100000,
        type=int
    )

    parser.add_argument(
        '--minSatElevation',
        default=30,
        type=int
    )

    parser.add_argument(
        '--linkingMethod',
        default=30,
        type=int
    )

    parser.add_argument(
        '--ecc',
        default=0.0,
        type=float
    )

    parser.add_argument(
        '--arcOfAscendingNodes',
        default=360.0,
        type=float
    )

    parser.add_argument(
        '--cmin_time',
        default=2,
        type=int
    )

    parser.add_argument(
        '--cmax_time',
        default=5,
        type=int
    )

    parser.add_argument(
        '--time_solt',
        type = float,
        default = 1.0
    )

    parser.add_argument(
        '--init_time',
        type=float,
        default=1.0
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
