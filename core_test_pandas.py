from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, A2C
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv

from TrainForexEnv import ForexTrainEnv

from datetime import datetime
import os

import time

model_class = TD3
model_name = 'TD3'

brain_name = 'A2C brain - 08 08 2023, 23-07-48'
brain_checkpoint = 'checkpoint 10000 - 20000'
brain_replay = 'replay 30000 - 40000'

model_info = {'model_name': model_name,
            'brain_name' : brain_name,
            'brain_checkpoint' : brain_checkpoint,
             'brain_replay' : brain_replay,}

wallet = 135
max_wallet = wallet

reward_list = []

for i in range(30):

    env = ForexTrainEnv(wallet,"EURJPY",
                        ticks_before=10,
                        brain_name=brain_name,
                        random_money=False,
                        begin_date='1/1/2023 06:00:00',
                        end_date='1/1/2024 06:00:00' )

    brain_path = f"C:\\Users\\sonic\\OneDrive\\Escritorio\\GAINSRL\\modelos\\TD3\\TD3 - 05 02 2024, 06-38-13\\gen 2\\checkpoint TD3 brain - 05 02 2024, 06-38-13 - ag 3.zip"
    model = model_class.load(brain_path,env)

    # replay_path = f"..\\models\\{model_info['model_name']}\\{model_info['brain_name']}\\{model_info['brain_replay']}.pkl"
    # model.load_replay_buffer(replay_path)

    obs = env.reset()

    while True:
        #time.sleep(1)
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, info = env.step(action)

        wallet = info["wallet"]

        if wallet > max_wallet:
            max_wallet = wallet

        diff = max_wallet - wallet

        # if diff > max_wallet*0.04:
        #     max_wallet = wallet
        #     break
        
        if terminated:
            obs = env.reset()
            reward_list.append(env.reward_total)
            break

print(reward_list)
print(sum(reward_list))
 


