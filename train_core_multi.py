from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, A2C, PPO
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np

from TrainForexEnv import ForexTrainEnv

from datetime import datetime
import os

import argparse
import sys

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-f", "--fecha", help="Your input file.")
    parser.add_argument("-d", "--directorio", help="Your destination output file.")
    parser.add_argument("-a", "--agente", help="Your destination output file.")
    parser.add_argument("-m", "--modelo", help="Your destination output file.")
    parser.add_argument("-b", "--checkpoint_brain", help="Your destination output file.")
    parser.add_argument("-k", "--checkpoint_pkl", help="Your destination output file.")
    parser.add_argument("-g", "--generacion", help="Your destination output file.")
    options = parser.parse_args(args)
    return options


options = getOptions(sys.argv[1:])

model_name = options.modelo
model_class = getattr(sys.modules[__name__], model_name)  # works also with SAC, DDPG and TD3

date_time = options.fecha

brain_name = f'{model_name} brain - {date_time} - ag {options.agente}'

log_name = f'{brain_name} - {options.generacion}'

checkpoints_path = options.directorio

model_info = {'name': model_name,
            'brain_name' : brain_name,
            'checkpoints_path' : checkpoints_path,
            'path_load': options.checkpoint_brain,
            'replay_load':options.checkpoint_pkl,
}



# brain_name = 'test'
# model_class = TD3
# log_name = "meh"

logdir = "logs_GEN"

env = ForexTrainEnv(123,"EURUSD",
                    ticks_before=10,
                    brain_name=brain_name,
                    random_money=True,
                    begin_date='1/1/2020 06:00:00',
                    end_date='1/1/2022 06:00:00' )

try:
    model = model_class.load(model_info['path_load'],env)
    model.load_replay_buffer(model_info['replay_load'])
    print(f"Modelo {model_info['path_load']} {model_name} se cargo")
except:
    action_noise = NormalActionNoise(mean=np.zeros(3), sigma=3 * np.ones(3))
    model = model_class("MlpPolicy",env,verbose = 1, tensorboard_log=logdir,learning_starts=10000,buffer_size=80000,action_noise= action_noise)
    print(f'Modelo A2C nuevo Creado')

#print(model.policy)

#Total de pasos por hacer
TIMESTEPS = 11000

CHECKPOINTS = 1

TIMESTEPS_PER_CHECKPOINT = TIMESTEPS // CHECKPOINTS

EPISODIES_PER_LOG = 1

for i in range(CHECKPOINTS):

    model.learn(total_timesteps=TIMESTEPS_PER_CHECKPOINT, reset_num_timesteps=False, log_interval=EPISODIES_PER_LOG, tb_log_name=log_name)
    #Guardar el checkpoint del modelo
    print("saving the model...")
    save_path = model_info['checkpoints_path']+"\\checkpoint "+model_info['brain_name']
    model.save(save_path)

    print("Saving buffer replay ")
    replay_path = model_info['checkpoints_path']+"\\replay "+model_info['brain_name']
    model.save_replay_buffer(replay_path)

env.write_report(model_info['checkpoints_path']+"\\report "+model_info['brain_name'], options.agente )