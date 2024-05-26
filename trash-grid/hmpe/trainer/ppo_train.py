from tqdm import tqdm
import numpy as np


astr = ['Forward','left','right','pickup','splitb','putdown']

def ppo_train(env, policy, num_episodes, logger=None):
    return_list = []
    results = {'agent_0': [], 'agent_1': [], 'agent_2': []}
    loss = {'actor': [], 'critic': []}
    with open(logger.log_dir + '/out.txt', 'w') as f:
        for iter in range(num_episodes // 8):
            with tqdm(total=int(8), desc='Iteration %d' % iter) as pbar:
                rews = []
                states = {}
                for a in env.n_agents:
                    states[a] = []
                for i_episode in range(10):
                    episode_return = 0
                    t_data = {}
                    for agent in env.n_agents:
                        t_data[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': []}
                    # num_agent = env.num_agents
                    state, obss = env.reset()
                    if i_episode == 0:
                        f.write(f'Iter: {iter}\nInit-State:\n{state}\n')
                    # obss, _ = env.reset()
                    done = False
                    while not done:
                        for agent in env.n_agents:
                            action = policy.take_action(obss[agent])
                            state, reward, done, _ = env.step(agent, action)
                            t_data[agent]['obs'].append(obss[agent])
                            t_data[agent]['actions'].append(action)
                            t_data[agent]['rewards'].append(reward)
                            # t_data[agent]['dones'].append(done)
                            next_obss = env.updateobs()
                            t_data[agent]['next_obs'].append(next_obss[agent])
                            t_data[agent]['dones'].append(done)
                            episode_return += reward
                            if i_episode == 0:
                                states[agent].append(state)
                                f.write(f'{agent} action:{action}-{astr[action]}')
                                f.write(f"state:\n{state}\n")
                                f.write(f"obs: {obss[agent]}\n")
                                f.write(f"reward: {reward}\n")
                        next_obss = env.updateobs()
                        # for agent in env.n_agents:
                        #     t_data[agent]['next_obs'].append(next_obss[agent])
                        #     t_data[agent]['dones'].append(done)
                        obss = next_obss

                    # if i_episode == 0:
                    #     # f.write(f'iter: {iter}\n')
                    #     for agent in env.n_agents:
                    #         for i in range(len(t_data[agent]['obs'])):
                    #             f.write(str(states[agent][i]) + '\n')

                    #             f.write(str(t_data[agent]['obs'][i]) + '\n')
                    #             f.write('action: ' + str(t_data[agent]['actions'][i]) + '  ' +  str(astr[t_data[agent]['actions'][i]]) + '\n')
                    #             f.write('rewards: ' + str(t_data[agent]['rewards'][i]) + '\n')
                    return_list.append(episode_return)
                    rews.append(episode_return)
                    for agent in env.n_agents:
                        # results[agent].append(t_data[agent])
                        actor_loss, critic_loss = policy.update(t_data[agent])
                        loss['actor'].append(actor_loss)
                        loss['critic'].append(critic_loss)

                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (8 * iter + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)

                if logger:
                    logger.add_scalar('actor-loss', np.mean(loss['actor']), iter)
                    logger.add_scalar('reward', np.mean(rews), iter)
                    logger.add_scalar('critic-loss', np.mean(loss['critic']), iter)
                loss['actor'] = loss['critic'] = []
    return loss, results