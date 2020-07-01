import argparse

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
from models.conv_to_fc_net import ConvToFCNet

from tests import test_rollout

from rollout import Controller
import os

import torch

import ray

def average(lst):
    return sum(lst) / len(lst)


harvest_default_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    'entropy_coeff': -.000687}



def main(args):
    ray.init()
    controller = Controller()
    path = os.path.abspath(os.path.dirname(__file__)) # just saves in current directory right now

    epochs = 200
    horizon_len = 1000
    train_every = 50
    print_every = 1

    for epoch in range(epochs):
        rewards, observations, full_obs = controller.rollout(horizon_len, train_every=train_every, save_path=None)

        if epoch % print_every == 0:
            print("Epoch: {}".format(epoch))
            print(rewards)
            print("Average reward: {}".format(average(rewards)))
            # for i in range(controller.num_agents):
                # agent_i = "agent-{}".format(i)
                # print(rewards[i])


    # visualization
    controller.render_rollout(horizon=horizon_len, path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SSD/RL Stuff")
    parser.add_argument("--exp_name", type=str, default=None, help="Name of the ray_results experiment directory where results are stored.")
    parser.add_argument("--env", type=str, default="harvest", help="Name of the environment to rollout. Can be cleanup or harvest.")
    parser.add_argument("--algorithm", type=str, default="A3C", help="Name of the rllib algorithm to use.")
    parser.add_argument("--num_agents", type=int, default="5", help="Number of agent policies")
    parser.add_argument("--train_batch_size", type=int, default="30000", help="Size of the total dataset over which one epoch is computed.")
    parser.add_argument("--checkpoint_frequency", type=int, default="20", help="Number of steps before a checkpoint is saved.")
    parser.add_argument("--training_iterations", type=int, default="10000", help="Total number of steps to train for")
    parser.add_argument("--num_cpus", type=int, default="2", help="Number of available CPUs")
    parser.add_argument("--num_gpus", type=int, default="1", help="Number of available GPUs")
    parser.add_argument("--use_gpus_for_workers", action="store_true", help="Set to true to run workers on GPUs rather than CPUs")
    parser.add_argument("--use_gpu_for_driver", action="store_true", help="Set to true to run driver on GPU rather than CPU.")
    parser.add_argument("--num_workers_per_device", type=int, default="2", help="Number of workers to place on a single device (CPU or GPU)")

    args = parser.parse_args()

    main(args)
