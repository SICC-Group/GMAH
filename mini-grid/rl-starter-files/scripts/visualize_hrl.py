import argparse
import numpy
import os

import utils
from utils import device
from gymnasium import spaces

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--high_model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

parser.add_argument("--low_model", default=None,
                    help="name of the lowlevel model")
parser.add_argument("--goal_space", type=int, default=3,
                    help="subgoal space:[0,1,2]")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment
# env = utils.make_env(args.env, 100, args.seed, render_mode="human")
env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

lmodel_dir = utils.get_model_dir(args.low_model)
lagent = utils.Agent(env.observation_space, env.action_space, lmodel_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
goal_space = spaces.Discrete(args.goal_space)
hmodel_dir = utils.get_model_dir(args.high_model)
hagent = utils.Agent(env.observation_space, goal_space, hmodel_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []
    # gifs = []

# Create a window to view the environment
env.render()

missionspace = [
            "pick the key",
            "open the door",
            "get to the goal"
        ]

for episode in range(args.episodes):
    obs, _ = env.reset()
    print(f'Episode[{episode}/{args.episodes}]----Goal: {env.realgoal}, Mission: {env.mission}')
    while True:
        # env.render()
        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))
        mission = hagent.get_action(obs)
        obs['mission'] = missionspace[mission]
        print(missionspace[mission])
        action = lagent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        lagent.analyze_feedback(reward, done)
        # print(action)
        if done:
            break
    # env.render()
    frames.append(numpy.moveaxis(env.get_frame(), 2, 0))
    if args.gif:
        print("Saving gif... ", end="")
        gifdir = os.path.join(hmodel_dir, 'visualize_gif')
        if not os.path.exists(gifdir):
            os.makedirs(gifdir)
        write_gif(numpy.array(frames), os.path.join(gifdir, args.gif+str(episode)+".gif"), fps=1/args.pause)
        print("Done.")
    frames = []
    print('Done')

