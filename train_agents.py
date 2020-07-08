import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.agents.a3c.a3c_tf_policy_graph import A3CPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
# import tensorflow as tf
import collections

import argparse

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
from models.conv_to_fc_net import ConvToFCNet

import sys

from typing import Dict
import numpy as np

# args = tf.app.args.args
#


def average(lst):
    return sum(lst) / len(lst)


harvest_default_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    'entropy_coeff': -.000687}

cleanup_default_params = {
    'lr_init': 0.00126,
    'lr_final': 0.000012,
    'entropy_coeff': -.00176}


def on_episode_start(info):
    episode = info["episode"]
    # print("episode {} started".format(episode.episode_id))
    # sys.stdout.flush()
    episode.policy_rewards = collections.defaultdict(list)

# def on_episode_step(info):
#     episode = info["episode"]
#     pole_angle = abs(episode.last_observation_for()[2])
#     episode.user_data["pole_angles"].append(pole_angle)

def on_episode_end(info):
    episode = info["episode"]
    n_agents = 0
    episode_rewards = []
    for (_, policy_id), reward in episode.agent_rewards.items():
        n_agents += 1
        episode.policy_rewards[policy_id].append(reward)
        episode_rewards.append(reward)
        print("agent-{}: {}".format(n_agents, reward))
    # print("episode {} ended with length {}".format(
    #     episode.episode_id, episode.length))
    # print(episode.policy_rewards)
    print("Sum Reward: {}".format(sum(episode_rewards)))
    print("Avg Reward: {}".format(average(episode_rewards)))
    print("Min Reward: {}".format(min(episode_rewards)))
    print("Max Reward: {}".format(max(episode_rewards)))
    # Gini coefficient calc
    sum_abs_diff = 0
    for i in range(n_agents):
        for j in range(n_agents):
            sum_abs_diff += np.abs(episode_rewards[i] - episode_rewards[j])
    gini_coeff = sum_abs_diff / (2 * n_agents * sum(episode_rewards))
    print("Gini Coefficient: {}".format(gini_coeff))
    # 20:20 ratio calc
    n_20 = max(1, int(np.round(n_agents / 5, 0)))
    sorted_rews = sorted(episode_rewards)
    min_20 = sum(sorted_rews[:n_20])
    max_20 = sum(sorted_rews[n_agents-n_20:])
    if min_20 == 0:
        print("20:20 Ratio: Undefined")
    else:
        ratio_20 = max_20 / min_20
        print("20:20 Ratio: {}".format(ratio_20))
    if sorted_rews[0] == 0:
        print("Max-min Ratio: Undefined")
    else:
        print("Max-min Ratio: {}".format(sorted_rews[-1] / sorted_rews[0]))


    # Need env.agents
    print("Extrinsic Rewards:")
    for agent in info["env"].agents.values():
        print(agent.extrinsic_reward_record)

    sys.stdout.flush()




def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1):

    if env == 'harvest':
        def env_creator(_):
            return HarvestEnv(num_agents=num_agents)
        single_env = HarvestEnv()
    else:
        def env_creator(_):
            return CleanupEnv(num_agents=num_agents)
        single_env = CleanupEnv()

    env_name = env + "_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        # return (PPOPolicyGraph, obs_space, act_space, {})
        # return (A3CPolicyGraph, obs_space, act_space, {})
        return (None, obs_space, act_space, {}) # should be default now

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    model_name = "conv_to_fc_net"
    ModelCatalog.register_custom_model(model_name, ConvToFCNet)

    agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
                "train_batch_size": train_batch_size,
                "horizon": 1000,
                "num_workers": num_workers,
                "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
                "num_cpus_for_driver": cpus_for_driver,
                "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
                "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
                "callbacks": {
                    "on_episode_start": tune.function(on_episode_start),
                    "on_episode_end": tune.function(on_episode_end),
                },
                # "multiagent": {
                #     "policy_graphs": policy_graphs,
                #     "policy_mapping_fn": tune.function(policy_mapping_fn),
                # },
                "model": {"custom_model": "conv_to_fc_net", "use_lstm": True,
                          "lstm_cell_size": 128}

    })

    if algorithm in ["A2C", "A3C"]:
        config.update({"lr_schedule":
                [[0, hparams['lr_init']],
                    [20000000, hparams['lr_final']]],
                       "entropy_coeff": hparams['entropy_coeff']
                       })


    return algorithm, env_name, config


def main(args):
    ray.init(num_cpus=args.num_cpus, redirect_output=True)
    if args.env == 'harvest':
        hparams = harvest_default_params
    else:
        hparams = cleanup_default_params
    alg_run, env_name, config = setup(args.env, hparams, args.algorithm,
                                      args.train_batch_size,
                                      args.num_cpus,
                                      args.num_gpus, args.num_agents,
                                      args.use_gpus_for_workers,
                                      args.use_gpu_for_driver,
                                      args.num_workers_per_device)

    if args.exp_name is None:
        exp_name = args.env + '_' + args.algorithm
    else:
        exp_name = args.exp_name
    print('Commencing experiment', exp_name)

    run_experiments({
        exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": args.training_iterations
            },
            'checkpoint_freq': args.checkpoint_frequency,
            "config": config,
        }
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SSD/RL Stuff")
    parser.add_argument("--exp_name", type=str, default=None, help="Name of the ray_results experiment directory where results are stored.")
    parser.add_argument("--env", type=str, default="harvest", help="Name of the environment to rollout. Can be cleanup or harvest.")
    parser.add_argument("--algorithm", type=str, default="A3C", help="Name of the rllib algorithm to use.")
    parser.add_argument("--num_agents", type=int, default="5", help="Number of agent policies")
    parser.add_argument("--train_batch_size", type=int, default="30000", help="Size of the total dataset over which one epoch is computed.")
    parser.add_argument("--checkpoint_frequency", type=int, default="20", help="Number of steps before a checkpoint is saved.")
    parser.add_argument("--training_iterations", type=int, default="500", help="Total number of steps to train for")
    parser.add_argument("--num_cpus", type=int, default="2", help="Number of available CPUs")
    parser.add_argument("--num_gpus", type=int, default="0", help="Number of available GPUs")
    parser.add_argument("--use_gpus_for_workers", action="store_true", help="Set to true to run workers on GPUs rather than CPUs")
    parser.add_argument("--use_gpu_for_driver", action="store_true", help="Set to true to run driver on GPU rather than CPU.")
    parser.add_argument("--num_workers_per_device", type=int, default="2", help="Number of workers to place on a single device (CPU or GPU)")

    args = parser.parse_args()

    main(args)
