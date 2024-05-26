from env.hmpe_v2 import hmpe_v2 as hmpe
env = hmpe()
states, obss = env.reset()
print(states)
print(obss)