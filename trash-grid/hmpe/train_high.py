from utils.utils import parsetoJson, get_dim_from_space
from trainer.high_train import high_train_policy as high_trainer
from trainer.high_train import high_eval_policy as eval_trainer
from trainer.high_train import high_train_smac_policy as high_smac_trainer
from trainer.high_train import high_eval_smac_policy as eval_smac_trainer
# import utils.utils.trainer

import os
import argparse
from typing import Optional, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import time
from algs.ppo_cnn import PPO

from torch.utils.tensorboard import SummaryWriter

from env.hmpe import hmpe
from env.starcraft2.StarCraft2_Env import StarCraft2Env
from env.starcraft2.smac_maps import get_map_params
from env.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    t = time.strftime("%m-%d-%H%M", time.localtime())
    parser.add_argument("--task", type=str, default=f'low-{t}')
    parser.add_argument("--seed", type=int, default=11)

    parser.add_argument("--alg_name", type=str, default='GMAH')
    parser.add_argument("--num_episodes_step", type=int, default=1000000)
    parser.add_argument("--max_cycles", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="a smaller gamma favors earlier win")
    parser.add_argument("--lmbda", type=float, default=0.95, help="a")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=48)
    parser.add_argument("--emb_size", type=int, default=4, help="ont_hot of goal space")
    parser.add_argument("--hidden_dim", type=int, nargs="*", default=64)
    parser.add_argument("--test_episodes", type=int, default=32)
    parser.add_argument("--log_interval", type=int, default=6000)
    parser.add_argument("--eval_interval", type=int, default=10000)

    parser.add_argument("--fix_interval", type=int, default=5)
    
        #train
    parser.add_argument('--n_rollout_threads', type=int,  default=1,
                        help="Number of parallel envs for training rollout")
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--use_eval", type=bool, default=False)
    #env
    parser.add_argument("--env_name", type=str, default='StarCraft2')
    parser.add_argument("--num_agents", type=int, default=3)

    #smac
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")
    parser.add_argument('--use_global_all_local_state', action='store_true',
                        default=True, help="Whether to use available actions")
    parser.add_argument("--use_text", type=bool, default=False)
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

    if args.env_name == 'hmpe':
        env = hmpe(max_cycles=args.max_cycles, is_goaltrain=args.use_text)
    else:
        env = make_train_env(args)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.env_name == 'hmpe':
        #env.reset(seed=seed)
        obs_dim = env.obs_dim
        share_obs_dim = env.state_dim
        action_dim = env.action_space
        goal_space = env.goal_space
        max_cycles = args.max_cycles
    else:
        obs_dim = get_dim_from_space(env.observation_space[0])
        action_dim = get_dim_from_space(env.action_space[0])
        share_obs_dim = get_dim_from_space(env.share_observation_space[0])
        goal_space = len(env._goal_space)
        max_cycles = get_map_params(args.map_name)["limit"]
    # print(obs_shape)
    # policy = []
    # for i in range(3):
    #     policy.append(PPO(obs_shape, action_dim, args.actor_lr, args.critic_lr, args.lmbda,
    #             args.epochs, args.eps_train, args.gamma, device,
    #             use_text=args.use_text,
    #             goal_space=goal_space,
    #             embedding_size=args.emb_size
    #             ))
    low_policy = PPO(obs_dim, share_obs_dim, args.hidden_dim, action_dim, args.actor_lr, args.critic_lr, args.lmbda,
                args.epochs, args.eps_train, args.gamma, device,
                use_text=True,
                goal_space=goal_space
                )
    
    high_policy = PPO(obs_dim, share_obs_dim, args.hidden_dim, goal_space, args.actor_lr, args.critic_lr, args.lmbda,
                args.epochs, args.eps_train, args.gamma, device,
                use_text=True,
                goal_space=goal_space,
                use_obs = args.use_obs_instead_of_state
                #args.use_obs_instead_of_state
                )

    # Train
    # TODO 间隔随机采样设置智能体的goal
    if args.env_name == 'hmpe':
        trainer = eval_trainer if args.use_eval else high_trainer
    else:
        trainer = eval_smac_trainer if args.use_eval else high_smac_trainer
    
    if args.env_name == 'hmpe':
        low_run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                        0] + "/results") / args.env_name / args.exp_name / "ppo-low"
        load_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                        0] + "/results") / args.env_name / args.exp_name / "ppo-high"

        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                        0] + "/results") / args.env_name / args.exp_name /"ppo-high"
    else:
        low_run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                        0] + "/results") / args.env_name / args.map_name / args.exp_name / "ppo-low"
        load_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                        0] + "/results") / args.env_name / args.map_name / args.exp_name / "ppo-high"

        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                        0] + "/results") / args.env_name / args.map_name / args.exp_name /"ppo-high"
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exist_run_nums = [int(str(folder.name).split('run')[
                                1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exist_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exist_run_nums) + 1)

    if not low_run_dir.exists():
        low_run = 'run1'
    else:
        exist_run_nums = [int(str(folder.name).split('run')[
                                1]) for folder in low_run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exist_run_nums) == 0:
            low_run = 'run1'
        else:
            low_run = 'run%i' % (max(exist_run_nums))
      
    save_dir = str(low_run_dir / low_run)
    #hmpe run18
    actor_dict = torch.load(save_dir + '/actor_policy.pth')
    critic_dict = torch.load(save_dir + '/critic_policy.pth')
    low_policy.actor.load_state_dict(actor_dict)
    low_policy.critic.load_state_dict(critic_dict)

    # save_high_dir = str(load_dir  / 'run47')
    # high_actor_dict = torch.load(save_high_dir + '/actor_policy.pth')
    # high_critic_dict = torch.load(save_high_dir + '/critic_policy.pth')
    # high_policy.actor.load_state_dict(high_actor_dict)
    # high_policy.critic.load_state_dict(high_critic_dict)
    
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if args.env_name == 'hmpe':
        progress_filename = os.path.join(run_dir,'progress.csv')
        df = pd.DataFrame(columns=['step','reward','num_put_trash','finish_time'])
        df.to_csv(progress_filename,index=False)

        progress_filename = os.path.join(run_dir,'progress_eval.csv')
        df = pd.DataFrame(columns=['step','reward','num_put_trash','finish_time'])
        df.to_csv(progress_filename,index=False)
    else:
        progress_filename = os.path.join(run_dir,'progress.csv')
        df = pd.DataFrame(columns=['step','reward','win_rate','death_enemy'])
        df.to_csv(progress_filename,index=False)

        progress_filename = os.path.join(run_dir,'progress_eval.csv')
        df = pd.DataFrame(columns=['step','reward','win_rate','death_enemy'])
        df.to_csv(progress_filename,index=False)
    
    if args.use_eval:
        loss = trainer(env, low_policy, high_policy, 0 , env.num_agents, max_cycles , run_dir , args.fix_interval, args.test_episodes, args.n_rollout_threads)
    else:
        loss = trainer(env, low_policy, high_policy, args.num_episodes_step, logger, env.num_agents, args.buffer_size, max_cycles, run_dir, args.fix_interval, args.test_episodes, args.n_rollout_threads, args.log_interval, args.eval_interval)

    actor_model_save_path = os.path.join(run_dir, 'actor_policy.pth')
    critic_model_save_path = os.path.join(run_dir, 'critic_policy.pth')

    torch.save(high_policy.actor.state_dict(), actor_model_save_path)
    torch.save(high_policy.critic.state_dict(), critic_model_save_path)