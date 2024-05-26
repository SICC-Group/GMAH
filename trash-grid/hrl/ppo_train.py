import os
import argparse
from typing import Optional, Tuple

from pettingzoo.mpe import simple_hmpe_v3

import torch
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
import rl_utils
from ppo import PPO

from torch.utils.tensorboard import SummaryWriter

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='062327')
    parser.add_argument("--seed", type=int, default=1626)

    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_cycles", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=4e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="a smaller gamma favors earlier win")
    parser.add_argument("--lmbda", type=float, default=0.95, help="a smaller gamma favors earlier win")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, nargs="*", default=128)
    parser.add_argument("--test-episodes", type=int, default=10)

    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=str, default='rgb_array')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    args = get_args()

    # =========Logger===============
    log_path = os.path.join(os.getcwd(), 'hmpe', 'ppo', args.task)
    logger = SummaryWriter(log_path)

    seed = args.seed
    device = torch.device(args.device)

    env = simple_hmpe_v3.env(max_cycles=args.max_cycles, render_mode=args.render)
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    state_dim = env.observation_space('agent_0').shape[0]
    action_dim = env.action_space('agent_0').n
    agent = PPO(state_dim, args.hidden_dim, action_dim, args.actor_lr, args.critic_lr, args.lmbda,
                args.epochs, args.eps_train, args.gamma, device)

    # Train
    return_list, loss = rl_utils.train_on_policy_agent(env, agent, args.num_episodes, logger)

    actor_model_save_path = os.path.join(log_path, 'actor_policy.pth')
    critic_model_save_path = os.path.join(log_path, 'critic_policy.pth')

    torch.save(agent.actor.state_dict(), actor_model_save_path)
    torch.save(agent.critic.state_dict(), critic_model_save_path)

    # test_env = simple_hmpe_v3.env(max_cycles=max_cycles, render_mode=render_mode)
    # result = test_policy(agent, test_env, 2)