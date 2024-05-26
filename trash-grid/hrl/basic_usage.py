from pettingzoo.mpe import simple_hmpe_v3

env = simple_hmpe_v3.env(max_cycles=3, render_mode="human")
env.reset(seed=42)

state_dim = env.observation_space('agent_0').shape[0]
action_dim = env.action_space('agent_0').n
print(env.state())
print(action_dim)

for agent in env.agent_iter():
    print(agent)
    observation, reward, termination, truncation, info = env.last()
    print(env.rewards[agent])
    # print(observation, reward, info)
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
    env.step(action)
    # print(env.observe(agent))
env.close()