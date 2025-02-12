# GMAH
subGoal-based Multi-Agent Hierarchical reinforcement learning method

### Install
1.Building Docker Images
```shell
cd GMAH-latest
docker build -t gmah .
```
2.Generate Container
```shell
docker run -it gmah bash
```
3.Install custom package
```shell
pip install -e .
```
4.Environments supported

- [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)
### Train
1.Trash-Grid
```shell
./train_hmpe.sh
```
2.smac

Replace map_name to train different maps
```shell
./train_smac.sh 
```
### plot
执行`./scripts/evaluate_low.py, ./scripts/evaluate_hrl.py`
1.Trash-Grid
```shell
python plot_hmpe.py
```
2.smac
```shell
python plot_smac.py
```
