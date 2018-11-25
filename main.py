#!/usr/bin/env python
# coding: utf-8


import gym
import time
import numpy as np
import multiple
import QN.QN as QN
import DoubleQN.DoubleQN as DoubleQN
from lib import plotting
import argparse
import os



def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  
    parser = argparse.ArgumentParser(description='Run Reinforcment Learning at an Office in Tsinghua University')
    parser.add_argument('--env', default='multiple_control-v0', help='Environment name')
    parser.add_argument('-o', '--output', default='multiple_QN_Eplus_OneDay_5action_random', help='Directory to save data to')
    parser.add_argument('--num', default=400, help='Number of Episodes')
    parser.add_argument('--memory', default=2000, help='max size of replay memory')
    parser.add_argument('--gamma', default=0.95, help='Discount Factor')
    parser.add_argument('--alpha', default=0.5, help='Constant step-size parameter')
    parser.add_argument('--epsilon', default=1, help='Epsilon greedy policy')
    parser.add_argument('--epsilon_min', default=0.0105, help='Smallest Epsilon that can get')
    parser.add_argument('--epsilon_decay', default=0.99, help='Epsilon decay after the number of episodes')
    parser.add_argument('--batch_size', default=32, help='Sampling batch size')
    parser.add_argument('--lr', default=0.001, help='Learning rate')
    # also most change the environment consider_energy to True
    parser.add_argument('--consider_energy', default=False, help='Whether take energy efficiency as consideration')


    args = parser.parse_args()

    output = get_output_folder(args.output, args.env)

  

    #create environment
    print(args.env)
    env = gym.make(args.env)
    env_evaluation = gym.make(args.env)
    ############### Q learning with Neural Network approximation and fixed target ################
    state_size = env.nS
    action_size = env.nA
    occupant_size = env.nO
    agents = []
    for i in range(occupant_size):
        agents.append(QN.QNAgent(state_size, action_size, float(args.gamma), float(args.lr)))

    stats = QN.q_learning(env, env_evaluation, agents, int(args.num), int(args.batch_size),
       float(args.epsilon), float(args.epsilon_min), float(args.epsilon_decay), args.consider_energy, output)
    plotting.plot_episode_stats(stats, smoothing_window=1)
    # for i in range(occupant_size):
    #     agents.append(QN.QNAgent(state_size, action_size, float(args.gamma), float(args.lr), i, output, True))
    # QN.evaluation(env, agents, args.consider_energy, output + "evaluation")
    ### change the environment to  _process_state_DDQN before use it

   

    ############### Double Q neural network Learning ################
    #### change the environment to  _process_state_DDQN before use it

    # state_size = env.nS
    # action_size = env.nA
    # agent = DoubleQNPER.DoubleQNPERAgent(state_size, action_size, float(args.gamma), float(args.lr), int(args.memory))
    # stats = DoubleQNPER.q_learning(env, agent, int(args.num), int(args.batch_size),
    #      float(args.epsilon), float(args.epsilon_min), float(args.epsilon_decay), output)
    # plotting.plot_episode_stats(stats, smoothing_window=1)

    #DoubleQNPER.evaluation(env, agent, output)

if __name__ == '__main__':
    main()
