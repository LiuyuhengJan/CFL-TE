from algorithm import test_a
import ppoconfig

all_args = ppoconfig.get_parser().parse_known_args()[0]

agent = test_a.PPO(all_args)

agent.load_model('ppo_model_random.pth')


def get_agent():
    all_args = ppoconfig.get_parser().parse_known_args()[0]

    agent = test_a.PPO(all_args)

    agent.load_model('ppo_model_random.pth')

    return agent