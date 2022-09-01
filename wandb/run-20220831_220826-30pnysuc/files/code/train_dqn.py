
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
import random, numpy, argparse, logging, os
from collections import namedtuple
import numpy as np
import datetime, math
import gym
from dqn import replay_buffer, DQN
from wandb_utils import init_wandb
from torch.utils.tensorboard import SummaryWriter

# Hyper Parameters
MAX_EPI=1000
MAX_STEP = 10000
SAVE_INTERVAL = 200
TARGET_UPDATE_INTERVAL = 20

BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 2000

GAMMA = 0.95
EPSILON = 0.05  # if not using epsilon scheduler, use a constant
EPSILON_START = 1.
EPSILON_END = 0.05
EPSILON_DECAY = 10000
LR = 1e-4    

def rollout(env, model, id, writer):
    r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
    log = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    print('\nCollecting experience...')
    total_step = 0
    for epi in range(MAX_EPI):
        s=env.reset()
        epi_r = 0
        epi_loss = 0
        for step in range(MAX_STEP):
            # env.render()
            total_step += 1
            a = model.choose_action(s)
            s_, r, done, info = env.step(a)
            r_buffer.add([s,s_,[a],[r],[done]])
            model.epsilon_scheduler.step(total_step)
            epi_r += r
            if total_step > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                sample = r_buffer.sample(BATCH_SIZE)
                loss = model.learn(sample)
                epi_loss += loss
            if done:
                break
            s = s_
        print('Ep: ', epi, '| Ep_r: ', epi_r, '| Steps: ', step, f'| Ep_Loss: {epi_loss:.4f}', )
        writer.add_scalar(f"charts/reward", epi_r, epi)
        writer.add_scalar(f"charts/loss", epi_loss, epi)
        writer.add_scalar(f"charts/episode length", step, epi)
        log.append([epi, epi_r, step])
        if epi % SAVE_INTERVAL == 0:
            model.save_model(model_path=f'trained_models/dqn{id}')
            # np.save('log/'+timestamp, log)              # learning rate

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    print(env.observation_space, env.action_space)
    model = DQN(env)
    num_models = 1
    args = {
        'wandb_project': 'ensemble_dqn',
        'wandb_entity': 'quantumiracle',
    }
    for i in range(num_models):
        run_name = f'dqn{i}'
        args['wandb_name'] = run_name
        writer = SummaryWriter(f"runs/{run_name}")
        init_wandb(args)
        rollout(env, model, i, writer)