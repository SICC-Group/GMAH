import pandas as pd
import os
import glob
import numpy as np
import sys
import matplotlib.pyplot as plt
import csv
import math
from pathlib import Path




path1 = r'trash-grid/results/StarCraft2/{}/{}/IPPO'
path2 = r'trash-grid/results/StarCraft2/{}/{}/MAPPO'
path3 = r'trash-grid/results/StarCraft2/{}/{}/ppo-high'
path4 = r'offpolicy/scripts/results/StarCraft2/{}/mqmix/{}'
path5 = r'offpolicy/scripts/results/StarCraft2/{}/mqtran/{}'

exp_name = "debug"
total_step=1000000
map_name = "2s3z"

def read_data(path, idx):
    csv_files = glob.glob(os.path.join(path, "progress_eval.csv"))
    step = []
    win_rate = []
    
    df = pd.read_csv(csv_files[0])
    step = np.append(step, np.array(df["step"]))
    win_rate = np.append(win_rate, np.array(df["win_rate"]))

    return step,win_rate

def print_draw(path,st,label_list,style,color):

    step = []
    win_rate = []

    path_f = [Path(path[i].format(map_name,exp_name)) for i in range(len(path))]
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
        eps_step,eps_win_rate = read_data(path_l[i],1)
        step.append(eps_step)
        win_rate.append(eps_win_rate)


    x_list = ['0']
    for i in range(1,6):
        x_list.append(str(0.2*i)+'M')
    len_s = len(st)
    plt.figure(1)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i in range(len(path)):
        plt.plot(step[i][:len_s], win_rate[i][:len_s], label=label_list[i], linestyle=style[i],color = color[i])

    plt.xticks(np.arange(0,total_step+1,total_step//5),x_list)
    plt.xlabel("step")
    plt.ylabel("win_rate")
    plt.grid(True,linestyle='--',color='grey',linewidth=1)
    plt.xlim(0,total_step+1)
    plt.ylim(0,1)
    plt.legend()


if __name__ == '__main__':


    path = [path1,path2,path3,path4,path5]
    label_list = ["IPPO","MAPPO","GMAH","QMIX","QTRAN"]
    style = ["dashed","dashed","dashed","dashed","dashed"]
    color = ["b","orange","purple","r","g"]
    st = np.arange(0,19)
    print_draw(path,st,label_list,style,color)

    plt.show()
