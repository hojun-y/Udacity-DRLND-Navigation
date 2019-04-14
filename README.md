# Udacity-DRLND-Navigation
Udacity Deep Reinforcement Learning Nanodegree - Navigation project

## About the Environment
The goal of the agent is to collect as many yellow bananas as possible while avoding the blue banans.<br>
The detailed environment attribytes are as follows:
### Observations
 - Vector observation of **37 dimensions**.
 - Ray perception + agent's velocity

### Actions
 - **4 Actions**
 - Each action is *move forward*, *move backward*, *turn left*, and *turn right*.

### Rewards
 - **+1** : Collect yellow banana
 - **-1** : Collect blue banana
 
## Installation

### 1. Install the Environment
Download the environment from the links below:
 - Linux: <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip">download link</a>
 - Windows(x86): <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip">download link</a>
 - Windows(x86_64): <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip">download link</a>
 - MacOS: <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip">download link</a>
 
 After downloading the environment, unzip it into the folder where the `train.py` is located.
 
 ### 2. Install the Dependencies
 Install some dependencies.
 #### If you're using `conda`:
  > ```
  > conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
  > conda install numpy scipy matplotlib 
  > ```
  
 #### Or using `pip`:
 > ```
 > # Linux or MacOS only:
 > pip install torch
 >
 > # Windows only:
 > pip install https://download.pytorch.org/whl/cu90/torch-1.0.1-cp36-cp36m-win-amd64.wml
 >
 > pip install torchvision
 > pip install numpy scipy matplotlib
 > ```
 ### 3. Install Unity ML-Agents
 You should install ml-agents **version 0.4**.<br>
 (Because of the API version of the environment, latest version is not compatible)
 ```
 pip install mlagents==0.4
 ```
 For more information about Unity ML-Agents, please visit the
 <a href="https://github.com/Unity-Technologies/ml-agents/blob/master/docs/">official documentation</a>.

## 2. Train the agent
execute `train.py` after activate the python environment.
```
python train.py
```

If needed, you can change the hyperparameters by modifying `config.py` before training.

## 3. Result(Reward Plot)
The environment was solved in about ~700 episodes.
<img src="/save/reward.png" width="85%">
