# GMAH
subGoal-based Multi-Agent Hierarchical reinforcement learning method
## Mini-Grid
单智能体环境，智能体需要找到绿色的目标格子
可以旋转,前进,捡钥匙,开门,激活物品等
### Install
```bash
# 分别
cd ./Minigrid
cd ./torch-ac
# 对每个执行
pip install -e .
```
### Structure
**./mini-grid**

Minigrid: 实验用到的环境
- `./Minigrid/minigrid/envs/doorkey.py`

torch-ac: 实验用到的算法, torch-ac/algos
- A2C, PPO, h_ppo: 就是gmah的高层策略，收集数据并更新模型参数
- GMAH的低层直接用ppo方法，在负责训练的脚本中参数不同

rl-starter-files:这里是具体的模型和训练、测试、可视化的脚本
- ./model.py: actor和critic模型. a2c, ppo, gmah都是ac架构的
- ./scripts: 训练,测试,可视化用的一些脚本
- ./storage: 效果还好的一些结果, 有log有tensorboardx的记录 一些还有对应的模型文件 *.py
  有些有跑出来的gif图
- visual.ipynb脚本用来渲染trash-grid环境
### Train


model_save: storege
训练低层: 算法也是ppo,但需要改一下doorkey.py环境里的一些参数,有个控制是否使用子目标的 ` self.realgoal= xselectxxx`
gmah-adapt和gmah-no都是h_ppo, 在h_ppo里注掉/加上对应的使用自适应的代码

example:
训练低层/训练ppo,这个要训练a2c直接把ppo改成a2c就行了
`python3 -m scripts.train --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model DoorKey-8x8 --save-interval 10 --frames 10000000`
训练gmah高层
`python3 -m scripts.train --algo h_ppo --env MiniGrid-DoorKey-8x8-v0 --model DoorKey-8x8-hppo --save-interval 10 --frames 1000000 --goal_space 3 --low_model DoorKey-8x8-goal2 --text`
### Test
执行`./scripts/evaluate_low.py, ./scripts/evaluate_hrl.py`


### visualize
**example**
可视化ppo/低层/a2c 改模型 和代码里子目标开关就行
`python3 -m scripts.visualize --env MiniGrid-DoorKey-8x8-v0 --model DoorKey-8x8-goal --episodes 10 --gif visul_gif --text`
可视化gmah高层
`python3 -m scripts.visualize_hrl --env MiniGrid-DoorKey-8x8-v0 --low_model DoorKey-8x8-goal-new --high_model DoorKey-8x8-hppo --episodes 1 --gif visul_gif --text`

手动控制
使用`./scripts/interact_visualize.py`

保存的一些模型都在storage里
有些跑出来的gif也在里面,就是可能文件名字和模型对的有点乱
## Trash-Grid
多智能体环境：3个智能体需要在有限的时间内回收垃圾，小垃圾可以直接携带，大垃圾需要进行压缩后才能携带，重量都为1，每个智能体的负重都是3

### Install
pip install mpe
pip install 


### Structure


bug: 

### Train



### Test




### visualize