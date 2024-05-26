# from env.hmpe import hmpe
# from hmpe_v0 import hmpe
from env.tutorialenv import CustomActionMaskedEnvironment
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    # env = hmpe()
    env = CustomActionMaskedEnvironment()

    parallel_api_test(env, num_cycles=100_0000)