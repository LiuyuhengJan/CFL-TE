import argparse

def get_parser():

    parser = argparse.ArgumentParser(description="Hyperparameters for Reinforcement Learning Algorithm")

    parser.add_argument('--num_e', type=int, default=10, help='Number of exploration episodes')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='Learning rate for the actor network')
    parser.add_argument('--critic_lr', type=float, default=1e-4, help='Learning rate for the critic network')
    parser.add_argument('--num_episodes', type=int, default=200, help='Number of training episodes')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of hidden layers')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lmbda', type=float, default=0.95, help='Lambda for GAE')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--eps', type=float, default=0.2, help='Epsilon for clipping in PPO')
    parser.add_argument('--d_model', type=int, default=32, help='Dimension of input embeddings')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Dimension of the feed-forward network')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')

    # Additional model components
    parser.add_argument('--conv_out_channels', type=int, default=64,
                        help='Number of output channels in convolutional layers')
    parser.add_argument('--gru_hidden_dim', type=int, default=128, help='Number of hidden units in GRU')
    parser.add_argument('--action_dim', type=int, default=26, help='Dimensionality of the action space')
    parser.add_argument('--in_channels', type=int, default=32,
                        help='Number of output channels in convolutional layers')
    parser.add_argument('--out_channels', type=int, default=128)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--test_id', type=str, default='testgetb_0726newbufferv1')
    return parser