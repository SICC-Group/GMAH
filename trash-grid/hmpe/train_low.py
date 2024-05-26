from utils.utils import train_on_policy_agent as ppo_trainer
from utils.utils import parsetoJson
from trainer.low_train import low_train_policy as low_trainer
from trainer.low_train import mlow_train_policy as mlow_trainer
# import utils.utils.trainer
from env.hmpe import hmpe

import os
import argparse
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
import time
from algs.ppo_cnn import PPO

from torch.utils.tensorboard import SummaryWriter

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    t = time.strftime("%m-%d-%H%M", time.localtime())
    parser.add_argument("--task", type=str, default=f'low-{t}')
    parser.add_argument("--seed", type=int, default=16)

    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--max_cycles", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="a smaller gamma favors earlier win")
    parser.add_argument("--lmbda", type=float, default=0.95, help="a")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--emb_size", type=int, default=4, help="ont_hot of goal space")
    parser.add_argument("--hidden-dim", type=int, nargs="*", default=128)
    parser.add_argument("--test-episodes", type=int, default=10)

    parser.add_argument("--use_text", type=bool, default=True)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser

    # parser.add_argument("--seed", type=int, default=1626)
    # parser.add_argument("--eps-test", type=float, default=0.05)
    # parser.add_argument("--eps-train", type=float, default=0.1)
    # parser.add_argument("--buffer-size", type=int, default=20000)
    # parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win")
    # parser.add_argument("--n-step", type=int, default=3)
    # parser.add_argument("--epoch", type=int, default=50)
    # parser.add_argument("--step-per-epoch", type=int, default=50000)
    # parser.add_argument("--step-per-collect", type=int, default=2000)
    # parser.add_argument('--repeat-per-collect', type=int, default=10)
    # parser.add_argument("--batch-size", type=int, default=64)
    # parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128])

    # # ppo special
    # parser.add_argument('--vf-coef', type=float, default=0.5)
    # parser.add_argument('--ent-coef', type=float, default=0.0)
    # parser.add_argument('--eps-clip', type=float, default=0.2)
    # parser.add_argument('--max-grad-norm', type=float, default=0.5)
    # parser.add_argument('--gae-lambda', type=float, default=0.95)
    # parser.add_argument('--rew-norm', type=int, default=0)
    # parser.add_argument('--norm-adv', type=int, default=0)
    # parser.add_argument('--recompute-adv', type=int, default=0)
    # parser.add_argument('--dual-clip', type=float, default=None)
    # parser.add_argument('--value-clip', type=int, default=0)

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    args = get_args()

    # =========Logger===============
    log_path = os.path.join(os.getcwd(), 'log', 'low_level', args.task)
    logger = SummaryWriter(log_path)
    parsetoJson(args, log_path)
    seed = args.seed
    device = torch.device(args.device)

    env = hmpe(max_cycles=args.max_cycles, is_goaltrain=args.use_text)

    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)

    obs_shape = env.obs_space
    action_dim = env.action_space
    goal_space = env.goal_space

    # print(obs_shape)
    policy = []
    for i in range(3):
        policy.append(PPO(obs_shape, action_dim, args.actor_lr, args.critic_lr, args.lmbda,
                args.epochs, args.eps_train, args.gamma, device,
                use_text=args.use_text,
                goal_space=goal_space,
                embedding_size=args.emb_size
                ))
    # policy = PPO(obs_shape, action_dim, args.actor_lr, args.critic_lr, args.lmbda,
    #             args.epochs, args.eps_train, args.gamma, device,
    #             use_text=args.use_text,
    #             goal_space=goal_space
    #             )

    # Train
    # TODO 间隔随机采样设置智能体的goal
    trainer = None
    if args.use_text:
        trainer = mlow_trainer
    else:
        trainer = ppo_trainer
    loss = trainer(env, policy, args.num_episodes, logger)

    # actor_model_save_path = os.path.join(log_path, 'actor_policy.pth')
    # critic_model_save_path = os.path.join(log_path, 'critic_policy.pth')

    # torch.save(policy.actor.state_dict(), actor_model_save_path)
    # torch.save(policy.critic.state_dict(), critic_model_save_path)