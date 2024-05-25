import argparse
import numpy
import os

import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.2,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

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

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []
    # gifs = []

# Create a window to view the environment
env.render()

missionspace = ['use the key', 'open the door', 'get to the goal', 'use the key to open the door and then get to the goal']
gifdir = os.path.join(model_dir, args.gif)
if not os.path.exists(gifdir):
    os.makedirs(gifdir)

for episode in range(args.episodes):
    obs, _ = env.reset()
    env.realgoal = 'goal'
    env.mission = 'get to the goal'
    print(f'Episode[{episode}/{args.episodes}]----Goal: {env.realgoal}, Mission: {env.mission}')
    cnt = 0
    if args.gif:
        frames.append(numpy.moveaxis(env.get_frame(), 2, 0))
        write_gif(numpy.array(frames), os.path.join(gifdir, args.gif+str(episode) + str(cnt) + ".gif"), fps=1/args.pause)
        frames = []
    while True:
        # env.render()

        cmd = input("Select a subgoal: 0:[use the key], 1:[open the door], 2:[go to the goal]")
        idx = int(cmd)
        obs['mission'] = missionspace[idx]

        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done) 

        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))
            write_gif(numpy.array(frames), os.path.join(gifdir, args.gif+str(episode) + str(cnt) + ".gif"), fps=1/args.pause)
            frames = []
        # print(action)
        cnt += 1
        if done:
            break
    # env.render()
    frames = []
    print('Done')

