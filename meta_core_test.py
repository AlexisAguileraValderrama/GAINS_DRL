from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, A2C
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv

from TrainForexEnv import ForexTrainEnv

from datetime import datetime
import os

model_class = A2C
model_name = 'A2C'

brain_name = 'A2C brain - 08 08 2023, 23-07-48'
brain_checkpoint = 'checkpoint 10000 - 20000'
# brain_replay = 'replay 490000 - 560000'

model_info = {'model_name': model_name,
            'brain_name' : brain_name,
            'brain_checkpoint' : brain_checkpoint}

wallet = 300

while True:

    env = ForexTrainEnv(wallet,"EURUSD",ticks_before=10,brain_name=brain_name)

    brain_path = f"..\\models\\{model_info['model_name']}\\{model_info['brain_name']}\\{model_info['brain_checkpoint']}"
    model = None

    model = A2C.load(brain_path,env)
    # replay_path = f"..\\models\\{model_info['model_name']}\\{model_info['brain_name']}\\{model_info['brain_replay']}.pkl"
    #model.load_replay_buffer(replay_path)

    obs = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, info = env.step(action)

        if info["abort"]:
            wallet = info["wallet"]
            break
        
        if terminated:
            obs = env.reset()
