import pandas as pd
import os
import glob
import numpy as np
import sys
import matplotlib.pyplot as plt
import csv
import math
from pathlib import Path




path1 = r'trash-grid/results/hmpe/{}/IPPO'
path2 = r'trash-grid/results/hmpe/{}/MAPPO'
path3 = r'trash-grid/results/hmpe/{}/ppo-high'
path4 = r'offpolicy/scripts/results/hmpe/mqmix/{}'
path5 = r'offpolicy/scripts/results/hmpe/mqtran/{}'

exp_name = "debug"
total_step=1000000

def read_data(path, idx):
    csv_files = glob.glob(os.path.join(path, "progress_eval.csv"))
    step1 = []
    reward = []
    
    df = pd.read_csv(csv_files[0])
    step1 = np.append(step1, np.array(df["step"]))
    reward = np.append(reward, np.array(df["reward"]))

    return step1,reward

def print_draw(path,st,label_list,style,color):

    step = []
    reward = []

    path_f = [Path(path[i].format(exp_name)) for i in range(len(path))]
    path_l = []
    
    for i in range(len(path)):
        exist_run_nums = [int(str(folder.name).split('run')[
                                1]) for folder in path_f[i].iterdir() if str(folder.name).startswith('run')]
        if len(exist_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exist_run_nums))
        path_l.append(str(path_f[i] / curr_run))

    for i in range(len(path)):
        eps_step,eps_reward = read_data(path_l[i],1)
        step.append(eps_step)
        reward.append(eps_reward)


    x_list = ['0']
    for i in range(1,6):
        x_list.append(str(0.2*i)+'M')
    len_s = len(st)
    T_ = 80
    plt.figure(1)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i in range(len(path)):
        plt.plot(step[i][:len_s], reward[i][:len_s], label=label_list[i], linestyle=style[i],color = color[i])

    plt.xticks(np.arange(0,total_step+1,total_step//5),x_list)
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.grid(True,linestyle='--',color='grey',linewidth=1)
    plt.xlim(0,total_step+1)
    plt.legend()


if __name__ == '__main__':


    path = [path1,path2,path3,path4,path5]
    label_list = ["IPPO","MAPPO","GMAH","QMIX","QTRAN"]
    style = ["dashed","dashed","dashed","dashed","dashed"]
    color = ["b","orange","purple","r","g"]
    st = np.arange(0,19)
    print_draw(path,st,label_list,style,color)

    plt.show()
