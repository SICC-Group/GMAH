import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import argparse
from ppotrain.normalization import Normalization, RewardScaling
from ppotrain.replaybuffer import ReplayBuffer
from ppotrain.ppo_discrete import PPO_discrete

import time
from utils.utils import parsetoJson
from env.hmpe_v2 import hmpe_v2

astr = ['Forward', 'Left', 'Right', 'Pickup', 'Split', 'Putdown']

def evaluate_policy(args, env:hmpe_v2, policy, state_norm, logdir=None):
    times = 10
    evaluate_reward = 0
    f = open(logdir + '/out.txt', 'w')
    for _ in range(times):
        s, obs = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            for a in env.n_agents:
                obs[a] = state_norm(obs[a], update=False)
        done = False
        episode_reward = 0
        f.write(f'[Iter/{_}]=================================')
        t = 0
        while not done:
            for i, agent in enumerate(env.n_agents):
                # TODO if get_g
                # f.write(f'[{agent}]\n')
                f.write(f"={agent}=[State]\n{env.states}\n")
                a = policy.evaluate(obs[agent])  # We use the deterministic policy during the evaluating
                s_, r, _, done, _ = env.step(agent, a)
                # if not args.use_state_norm:
                #     f.write(f"[Cur_Obs]\n{obs[agent]}\n")
                f.write(f"[action]: {astr[a]}\n")
                obs_ = env.updateobs()
                if args.use_state_norm:
                    obs_[agent] = state_norm(obs_[agent], update=False)
                episode_reward += r
                obs = obs_
                f.write(f"[Next State]\n{s_}\n")
                # if not args.use_state_norm:
                #     f.write(f"[Next Obs]\n{obs[agent]}\n")
                f.write(f"[reward]\t{r}\n")
            t += 1
        f.write(f'Episode Step:\t{t}\n')
        evaluate_reward += episode_reward
    f.flush()
    return evaluate_reward / times


def main(args, alg, seed, env_name):
    env = hmpe_v2(max_cycles=args.max_episode_steps, is_goaltrain=False)
    env_evaluate = hmpe_v2(max_cycles=args.max_episode_steps, is_goaltrain=False)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    device = torch.device(args.device)
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space + env.goal_space
    args.action_dim = env.action_space
    # args.max_episode_steps = 100  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = [ReplayBuffer(args) for _ in range(env.num_agents)]

    # 每个智能体1个策略网络？？
    # policy = [PPO_discrete(args) for _ in range(env.num_agents)]
    policy = PPO_discrete(args)

    t = time.strftime("%m-%d-%H%M", time.localtime())
    writer = SummaryWriter(log_dir='ppotrain/runs/{}_{}_{}_{}'.format(env_name, alg, seed, t))
    parsetoJson(args, writer.log_dir)

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    cunt = 0
    tau_shape = (3, args.max_episode_steps, 2)
    
    taus = []
    while total_steps < args.max_train_steps:
        s, obs = env.reset()
        if args.use_state_norm:
            for agent in env.n_agents:
                obs[agent] = state_norm(obs[agent])
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = [0] * env.num_agents
        done = False
        # record trajectory
        tau = np.zeros(tau_shape)
        tau.fill(-1)
        episode_reward = 0
        while not done:
            # total_steps += 1
            for i, agent in enumerate(env.n_agents):

                a, a_logprob = policy.choose_action(obs[agent])  # Action and the corresponding log probability
                s_, r, _, done, _ = env.step(agent, a)
                if r > 0:
                    print(agent, done, s, s_, r)
                obs_ = env.updateobs()
                episode_reward += r
                # Record trajectory
                # if cunt % 10 == 0:
                #     tau[i,episode_steps[i],:] = env.n_agents[agent].row, env.n_agents[agent].col
                # obs_ = obs_[agent]
                if args.use_state_norm:
                    for agent in env.n_agents:
                        obs_[agent] = state_norm(obs_[agent])
                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)
                episode_steps[i] += 1
                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if (done and episode_steps[i] != args.max_episode_steps):
                    dw = True
                else:
                    dw = False

                replay_buffer[i].store(obs[agent], a, a_logprob, r, obs_[agent], dw, done)
                obs = obs_
                s = s_
                # total_steps += 1

                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer[i].count == args.batch_size:
                    # policy[i].update(replay_buffer[i], total_steps)
                    policy.update(replay_buffer[i], total_steps)
                    replay_buffer[i].count = 0
            done = done
            # print(done, total_steps)
            total_steps += 1
            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, policy, state_norm, writer.log_dir)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./ppotrain/data_train/{}_{}_{}_{}.npy'.format(env_name, alg, seed, t), np.array(evaluate_rewards))
        cunt += 1
        writer.add_scalar('episode_rewards_{}'.format(env_name), episode_reward, global_step=total_steps)
        episode_reward = 0
        # taus.append(tau)
    # np.save('ppotrain/runs/{}_{}_{}_{}/tau.npy'.format(env_name, alg, seed, t), taus)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--task_name", type=str, default='mappo', help=" Maximum number of training steps")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--max_episode_steps", type=int, default=int(256), help=" Maximum number of episode steps")
    parser.add_argument("--evaluate_freq", type=float, default=4e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.02, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # main(args, env_name=env_name[env_index], number=1, seed=0)
    main(args, alg='mappo', seed=1, env_name='trash')