# Tennis collaboration.

This package offer policy based model-free methods to solve reinforcement learning  multi-agent environments with discrete or continuous action spaces.
Those environments are challenging to solve because of the non-sationarity of the environment from the point of view of every single agent.  
The intent of this implementation is to ease the experimentations by allowing the user to test various options using the modularized configuration of the agents and to store those respective configurations alongside their corresponding networks and performances.

The most successful implementation of this package is a take on [Multi Agent Deep Deterministic Policy Gradients](https://arxiv.org/abs/1706.02275) using PyTorch.

The experimental environment is a modified version of the tennis [Unity environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#tennis).

[![Python badge](https://img.shields.io/badge/python-v3.6-blue.svg)](https://shields.io/)

## The Environnement

In this environment, **two agents control rackets to bounce a ball over a net**. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The **observation space** consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. **Two continuous actions are available**, corresponding to movement toward (or away from) the net, and jumping.

The **task is episodic**, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically, after each episode, the rewards that each agent received (without discounting) is added up, to get a score for each agent. This yields 2 (potentially different) scores, the maximum of these 2 scores is kept as a single score for each episode.

The environment is considered **solved**, when the average (over 100 episodes) of those scores is at least **+0.5**.

## Getting started

1. Download the environment from one of the links below. You need to only select the environment that matches your operating sytem: 
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   - Max OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
 (*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
 
 (*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), the please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Once the environment is downloaded, you can rub the script `install.sh` on your virtual environment to install python dependencies and create needed directories:
- Trained model weights can be found in the `./output/model` directory.
- The global and by agent training socre are recoreded in the `./output/score` directory.
- The configurations used for training are also stored at `./output/configuration`.
- The environment folder at `./environment`.

3. Add the downloaded environment to the newly created folder `./environment`.

## Instructions

Follow the instructions in `Tennis_collaboration.ipynb` to train an agent an watch it interact with the environment.  
You will need to intanciate the `MADDPG_agents` with its `MADDPG_configuration` and make it learn by interacting with the environment throught the use a `Runner`.

**1. instantiate an agent :**

```python
from marl_coop.utils import Context
from marl_coop.config import MADDPG_configuration
from marl_coop.agent import MADDPG_agents

agent_cfg = {
    'seed':2,
    'gamma':.98,
    'tau':1e-3,
    'batch_size': 512,
    'update_every': 20,
    'learing_per_update': 5,
    'use_gradient_clipping': .8,
    'critic':{
        'criterion': 'MSE',
        'hidden_layers': [64, 54],
        'learning_rate': 3e-4
    },
    'actor':{
        'criterion': 'SmoothL1',
        'hidden_layers': [64, 54],
        'learning_rate': 8e-5
    },
    'buffer':{
        'type': 'prioritized_sumTree',
        'size': 100_000,
        'alpha': .8,
        'beta': .5,
        'epsilon':.015        
    },
    'noise':{
        'method': 'scaling_OU',
        'theta': .1,
        'theta_max': .3,
        'theta_nbr_growths': 100_000,
        'sigma': 1,
        'sigma_min': .009,
        'sigma_nbr_decays': 100_000
    }
}
agents_config={
    'agent_1': agent_cfg,
    'agent_2': agent_cfg
}
ctx = Context(brain, {'nbr_agents': nbr_agents, 'action_min': -1,  'action_max': 1})

cfg = MADDPG_configuration(nbr_agents=nbr_agents, agents_config=agents_config)
maddpg = MADDPG_agents(ctx, cfg)
```

**2. Make it learn by using a runner**

```python
from marl_coop.utils import Runner

key = 'my_key' # used to store the score / configuration / networks
nb_episodes = 8_000
runner = Runner()

runner.run(maddpg, env, brain_name, nb_episodes, key, target_score=.5,
               save_score=True, save_config=True, save_weights=True)
```

**3. Observe its performances during training**

```python
from marl_coop.utils import load_scores, plot_scores

score = load_scores(key='my_key')
plot_scores(score, window_size=100,target_score=.5)
```

**4. Watch it interact with the environment**

```python
env_info = env.reset(train_mode=False)[brain_name]    
observations = env_info.vector_observations           
scores = np.zeros(nbr_agents)                         

while True:
    
    actions = maddpg.act(observations, noise=True)
    
    env_info = env.step(actions)[brain_name]        
    next_observations = env_info.vector_observations
    rewards = env_info.rewards                      
    dones = env_info.local_done                     
    scores += env_info.rewards                      
    
    maddpg.step(observations, actions, rewards, next_observations, np.array(dones).reshape(-1,1))
    observations = next_observations                
    
    if np.any(dones):                               
        break
```

**It is also possible to load a pre-trained agent using the following method:**

```python
from marl_coop.utils.loader import load_MADDPG_agent

key = 'pre_trained'
maddpg = load_MADDPG_agent(key, ctx)
```

## Dependencies
1. Python 3.6
2. Pytorch
3. Unity ML-Agents

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Sources

- Original MADDPG paper by OpenAI [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).

- Udacity learning materials.