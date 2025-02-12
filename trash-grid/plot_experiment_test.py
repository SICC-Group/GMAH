import pandas as pd
import os
import glob
import numpy as np
import sys
import matplotlib.pyplot as plt
import csv
import math




path1 = r'trash-grid/results/StarCraft2/3s5z/debug/ppo-high/run2/'
path2 = r'trash-grid/results/StarCraft2/3s5z/debug/ppo-high/run3/'
# path3 = r'trash-grid/results/ppo-high/run113/'

#path1 = r'trash-grid/results/ppo-low/run18/'
# path2 = r'trash-grid/results/ppo-low/run18/'
# path3 = r'trash-grid/results/ppo-low/run18/'


def read_data(path, idx):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    step1 = []
    reward = []
    num_put_trash = []
    finish_time = []
    
    df = pd.read_csv(csv_files[idx])
    step1 = np.append(step1, np.array(df["step"]))
    reward = np.append(reward, np.array(df["reward"]))
    num_put_trash = np.append(num_put_trash, np.array(df["num_put_trash"]))
    finish_time = np.append(finish_time, np.array(df["finish_time"]))

    return step1,reward,num_put_trash,finish_time

def print_draw(path,st,label_list,style,color):

    step = []
    reward = []
    num_put_trash = []
    finish_time = []

    for i in range(len(path)):
        eps_step,eps_reward,eps_num_put_trash,eps_finish_time = read_data(path[i],1)
        step.append(eps_step)
        reward.append(eps_reward)
        num_put_trash.append(eps_num_put_trash)
        finish_time.append(eps_finish_time)
    # num_target.append(eps_num_target)
    # num_collision.append(eps_num_collision)

    # r_mean = []
    # target_mean = []
    # collision_mean = []
    # for i in range(len(path)):
    #     r_mean.append(np.zeros((len(step[i]))))
    #     target_mean.append(np.zeros((len(step[i]))))
    #     collision_mean.append(np.zeros((len(step[i]))))

    # for i in range(len(path)):
    #     for j in range(len(step[i])):
    #         if j>=3:
    #             r_mean[i][j] = np.mean(reward[i][j-3:j+3],axis=0)
    #             target_mean[i][j] = np.mean(num_target[i][j-3:j+3],axis=0)
    #             collision_mean[i][j] = np.mean(num_collision[i][j-3:j+3],axis=0)
    #         else:
    #             r_mean[i][j] = np.mean(reward[i][0:j+3],axis=0)
    #             target_mean[i][j] = np.mean(num_target[i][0:j+3],axis=0)
    #             collision_mean[i][j] = np.mean(num_collision[i][0:j+3],axis=0)

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
        plt.plot(st, reward[i][:len_s], label=label_list[i], linestyle=style[i],color = color[i])
    plt.xticks(np.arange(0,len(st)+2,(len(st)+1)//5),x_list)
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.grid(True,linestyle='--',color='grey',linewidth=1)
    plt.xlim(0,T_)
    plt.legend()

    plt.figure(2)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i in range(len(path)):
        plt.plot(st, num_put_trash[i][:len_s], label=label_list[i], linestyle=style[i],color = color[i])
    plt.xticks(np.arange(0,len(st)+2,(len(st)+1)//5),x_list)
    plt.xlabel("step")
    plt.ylabel("num_remain_trash")
    plt.grid(True,linestyle='--',color='grey',linewidth=1)
    plt.xlim(0,T_)
    plt.ylim(0,1)
    #plt.ylim(0,6)
    plt.legend()

    plt.figure(3)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i in range(len(path)):
        plt.plot(st, finish_time[i][:len_s], label=label_list[i], linestyle=style[i],color = color[i])
    plt.xticks(np.arange(0,len(st)+2,(len(st)+1)//5),x_list)
    plt.xlabel("step")
    plt.ylabel("finish_time")
    plt.grid(True,linestyle='--',color='grey',linewidth=1)
    plt.xlim(0,T_)
    plt.legend()

if __name__ == '__main__':


    # path = [path1,path2,path3,path4,path5,path6]
    # label_list = ["MAPPO","FGPPO","FGPPO-LOW-RANK","MAPPO-w/o locate","FGPPO-w/o locate","FGPPO-LR-w/o locate"]
    # style = ["solid","solid","solid","dashed","dashed","dashed"]
    # color = ["b","y","g","b","y","g"]
    path = [path1,path2]
    label_list = ["GMAH-high","GMAH-high-1"]
    #["FGPPO-range=0.5-rank=1","FGPPO-range=0.5-rank=3","FGPPO-range=0.8-rank=1","DDFG=0.8-rank=1"]
    style = ["dashed","dashed"]
    #["dashed","dashed","dashed"]
    color = ["r","b"]
    #["r","b","y"]
    st = np.arange(1,80)
    print_draw(path,st,label_list,style,color)







    plt.show()
