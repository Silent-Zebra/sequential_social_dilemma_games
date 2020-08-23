import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
# from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
# from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.rllib.agents.impala.vtrace_policy import VTraceTFPolicy


from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
# import tensorflow as tf
import collections

import ast

import argparse

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.constants import HARVEST_MAP, HARVEST_MAP_BIG, \
    HARVEST_MAP_TINY, HARVEST_MAP_TOY, HARVEST_MAP_CPR, \
    CLEANUP_MAP, CLEANUP_MAP_SMALL
from social_dilemmas.envs.cleanup import CleanupEnv
from models.conv_to_fc_net import ConvToFCNet, ConvToFCNetLarge

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
    'entropy_coeff': .000687}

cleanup_default_params = {
    'lr_init': 0.00126,
    'lr_final': 0.000012,
    'entropy_coeff': .00176}


def on_episode_start(info):
    episode = info["episode"]
    # print("episode {} started".format(episode.episode_id))
    # sys.stdout.flush()
    episode.policy_rewards = collections.defaultdict(list)

# def on_episode_step(info):
#     episode = info["episode"]
#     pole_angle = abs(episode.last_observation_for()[2])
#     episode.user_data["pole_angles"].append(pole_angle)


def print_episode_stats(n_agents, episode_rewards):
    print("Sum Reward: {}".format(sum(episode_rewards)))
    print("Avg Reward: {}".format(average(episode_rewards)))
    print("Min Reward: {}".format(min(episode_rewards)))
    print("Max Reward: {}".format(max(episode_rewards)))
    sys.stdout.flush()

    # Gini coefficient calc
    if sum(episode_rewards) == 0:
        print("Gini Coefficient: Undefined")
    else:
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
    max_20 = sum(sorted_rews[n_agents - n_20:])
    sys.stdout.flush()

    if min_20 == 0:
        print("20:20 Ratio: Undefined")
    else:
        ratio_20 = max_20 / min_20
        print("20:20 Ratio: {}".format(ratio_20))
    if sorted_rews[0] == 0:
        print("Max-min Ratio: Undefined")
    else:
        print("Max-min Ratio: {}".format(sorted_rews[-1] / sorted_rews[0]))

    sys.stdout.flush()



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

    # print_episode_stats(n_agents, episode_rewards)

    sys.stdout.flush()

    print("Extrinsic Rewards:")
    # print(info["env"].envs)
    # print(info["env"].envs[0])
    # print(info["env"].envs[0].agents)
    # print(info["env"].envs[0].agents.values())
    extrinsic_rewards = []

    for agent in info["env"].envs[0].agents.values():
        print(agent.extrinsic_reward_sum)
        extrinsic_rewards.append(agent.extrinsic_reward_sum)

    print_episode_stats(n_agents, extrinsic_rewards)

    sys.stdout.flush()

    print("Times Fired")
    for agent in info["env"].envs[0].agents.values():
        print(agent.fires)

    sys.stdout.flush()


    print("Times Hit")
    for agent in info["env"].envs[0].agents.values():
        print(agent.times_hit)

    # print("Times Clean")
    # for agent in info["env"].envs[0].agents.values():
    #     print(agent.cleans)

    # Use custom metrics if still not working
    # or consider
    # from pprint import pprint
    # pprint(vars(info["env"]))
    # pprint(vars(info["episode"]))
    # on info and info["env"] and info["episode"] to see what's available



    sys.stdout.flush()


def setup(env, hparams, algorithm, train_batch_size, rollout_fragment_length,
          training_iterations,
          num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1, num_envs_per_worker=1,
          # remote_worker_envs=False,
          custom_callback=True,
          intrinsic_rew_params=None,
          impala_replay=False,
          replay_proportion=0.0,
          replay_buffer_num_slots=0,
          conv_large=False,
          vtrace_policy=False,
          default_policy=False,
          harvest_map='regular',
          cleanup_map='regular', hit_penalty=50, fire_cost=1,
          gamma=0.99):

    if intrinsic_rew_params is None:
        ir_param_list = [None] * num_agents

    else:
        ir_param_list = intrinsic_rew_params.split(';')
        ir_param_list = list(map(ast.literal_eval, ir_param_list))
        assert len(ir_param_list) == num_agents

    if env == 'harvest':
        def env_creator(_):
            ascii_map = HARVEST_MAP
            if harvest_map == 'tiny':
                ascii_map = HARVEST_MAP_TINY
            elif harvest_map == 'toy':
                ascii_map = HARVEST_MAP_TOY
            elif harvest_map == 'cpr': # note only single agent at present
                ascii_map = HARVEST_MAP_CPR
            elif harvest_map == 'big':
                ascii_map = HARVEST_MAP_BIG
            created_env = HarvestEnv(ascii_map=ascii_map, num_agents=num_agents, ir_param_list=ir_param_list,
                              hit_penalty=hit_penalty, fire_cost=fire_cost)
            return created_env
        # example_env = env_creator()
    else:
        def env_creator(_):
            ascii_map = CLEANUP_MAP
            if cleanup_map == 'small':
                ascii_map = CLEANUP_MAP_SMALL
            created_env = CleanupEnv(ascii_map=ascii_map, num_agents=num_agents, ir_param_list=ir_param_list,
                              hit_penalty=hit_penalty, fire_cost=fire_cost)
            return created_env
    example_env = env_creator(None)

    env_name = env + "_env"
    register_env(env_name, env_creator)

    obs_space = example_env.observation_space
    act_space = example_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        # policy = A3CTFPolicy
        # if algorithm == "IMPALA":
        #     policy = VTraceTFPolicy
        # else:
        #     policy = A3CTFPolicy
        if vtrace_policy:
            policy = VTraceTFPolicy
        else:
            policy = A3CTFPolicy
        if default_policy:
            policy = None
        return (policy, obs_space, act_space, {})

    # Setup algorithm with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    if conv_large:
        model_name = "conv_to_fc_net_large"
        ModelCatalog.register_custom_model(model_name, ConvToFCNetLarge)
    else:
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

    horizon = 1000
    # hyperparams
    config.update({
                "train_batch_size": train_batch_size,
                "gamma": gamma,
                "horizon": horizon,
                "num_workers": num_workers,
                "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
                "num_cpus_for_driver": cpus_for_driver,
                "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
                "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
                "num_envs_per_worker": num_envs_per_worker,
                # "remote_worker_envs": remote_worker_envs,
                "sample_batch_size": rollout_fragment_length,

                "multiagent": {
                    "policies": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn),
                }
        # ,
                # "model": {"custom_model": "conv_to_fc_net", "use_lstm": True,
                #           "lstm_cell_size": 128}

    })

    if custom_callback:
        config.update({"callbacks": {
                    "on_episode_start": tune.function(on_episode_start),
                    "on_episode_end": tune.function(on_episode_end),
                }})


    if algorithm not in ["DQN"]:
        if conv_large:
            config.update(
                {"model": {"custom_model": "conv_to_fc_net_large", "use_lstm": True,
                           "lstm_cell_size": 256}})
        else:
            config.update({"model": {"custom_model": "conv_to_fc_net", "use_lstm": True,
                  "lstm_cell_size": 128}})

    if algorithm in ["A2C", "A3C", "IMPALA"]:
        config.update({"lr_schedule":
                [[0, hparams['lr_init']],
                    [training_iterations * horizon, hparams['lr_final']]],
                       "entropy_coeff": hparams['entropy_coeff']
                       })

    if algorithm in ["IMPALA"] and impala_replay:
        config.update({"replay_proportion": replay_proportion,
                       "replay_buffer_num_slots": replay_buffer_num_slots})



    return algorithm, env_name, config


def main(args):
    ray.init(num_cpus=args.num_cpus)
    if args.env == 'harvest':
        hparams = harvest_default_params
    else:
        hparams = cleanup_default_params
    if args.lr_init > 0.0:
        hparams['lr_init'] = args.lr_init
    if args.lr_final > 0.0:
        hparams['lr_final'] = args.lr_final
    if args.entropy_coeff > 0.0:
        hparams['entropy_coeff'] = args.entropy_coeff

    if args.no_custom_callback:
        custom_callback = False
    else:
        custom_callback = True

    conv_large=False
    if args.conv_large:
        conv_large=True

    alg_run, env_name, config = setup(args.env, hparams, args.algorithm,
                                      args.train_batch_size,
                                      args.rollout_fragment_length,
                                      args.training_iterations,
                                      args.num_cpus,
                                      args.num_gpus, args.num_agents,
                                      args.use_gpus_for_workers,
                                      args.use_gpu_for_driver,
                                      args.num_workers_per_device,
                                      args.num_envs_per_worker,
                                      # args.remote_worker_envs,
                                      custom_callback,
                                      args.intrinsic_rew_params,
                                      args.impala_replay,
                                      args.replay_proportion,
                                      args.replay_buffer_num_slots,
                                      conv_large,
                                      args.vtrace_policy,
                                      args.default_policy,
                                      args.harvest_map,
                                      args.cleanup_map,
                                      args.hit_penalty,
                                      args.fire_cost,
                                      args.gamma)

    if args.exp_name is None:
        exp_name = args.env + '_' + args.algorithm
    else:
        exp_name = args.exp_name
    print('Commencing experiment', exp_name)

    print(config)

    print("INTRINS REW PARAMS")
    print(args.intrinsic_rew_params)
    import sys
    sys.stdout.flush()

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
    # reduce batch size from 30k to 10k maybe esp given increased fragment length to 100
    parser.add_argument("--train_batch_size", type=int, default="10000", help="Size of the total dataset over which one epoch is computed.")
    parser.add_argument("--checkpoint_frequency", type=int, default="100", help="Number of steps before a checkpoint is saved.")
    parser.add_argument("--training_iterations", type=int, default="20000", help="Total number of steps (iters, not env steps) to train for")
    parser.add_argument("--num_cpus", type=int, default="2", help="Number of available CPUs")
    parser.add_argument("--num_gpus", type=int, default="0", help="Number of available GPUs")
    parser.add_argument("--use_gpus_for_workers", action="store_true", help="Set to true to run workers on GPUs rather than CPUs")
    parser.add_argument("--use_gpu_for_driver", action="store_true", help="Set to true to run driver on GPU rather than CPU.")
    parser.add_argument("--num_workers_per_device", type=int, default="2", help="Number of workers to place on a single device (CPU or GPU)")
    parser.add_argument("--num_envs_per_worker", type=int, default="1", help="For vectorized environment")
    # parser.add_argument("--remote_worker_envs", action="store_true", help="Ray's remote worker envs flag")

    # parser.add_argument("--intrinsic_rew_type", type=str, choices=['svo', 'ineq', 'altruism'], default=None,  help="Run agents with intrinsic reward modifications")
    parser.add_argument("--intrinsic_rew_params", type=str, default=None, help="Parameters for agents' intrinsic reward. Format: (rew_type, params) for each agent, semicolon delimited")
    # Example intrinsic reward params --intrinsic_rew_params "('ineq',5.0,0.05,0.01);('altruism',1.0,0.2,0.01);('svo',90,0.2,0.01);None;None"
    # Ineq aversion is alpha, beta
    # Altruism is w_self, w_others
    # SVO is angle (degrees), weight
    # THird param is intrinsic reward scaling (Effectively changing the learning rate)
    parser.add_argument("--harvest_map", type=str, default='regular', choices=['regular', 'tiny', 'big', 'toy', 'cpr'])
    parser.add_argument("--cleanup_map", type=str, default='regular', choices=['regular', 'small'])
    # parser.add_argument("--resume", action="store_true", help="Set to resume an experiment")
    parser.add_argument("--hit_penalty", type=int, default=50, help="Cost of being hit by a punishment beam")
    parser.add_argument("--fire_cost", type=int, default=1, help="Cost of firing a punishment beam")
    # parser.add_argument("--conv_small", action="store_true", help="Use smaller convnet architecture")
    parser.add_argument("--conv_large", action="store_true", help="Use larger convnet architecture")
    parser.add_argument("--no_custom_callback", action="store_true", help="No custom callback/metrics")
    parser.add_argument("--vtrace_policy", action="store_true", help="Use Vtrace policy base for IMPALA")
    parser.add_argument("--impala_replay", action="store_true", help="Use IMPALA Replay Buffer")
    parser.add_argument("--replay_proportion", type=float, default=0.5, help="Only if using IMPALA replay buffer")
    parser.add_argument("--replay_buffer_num_slots", type=int, default=10000, help="Only if using IMPALA replay buffer")
    parser.add_argument("--default_policy", action="store_true", help="Use None policy (base algorithm) - for testing")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor to use for the experiment")
    parser.add_argument("--rollout_fragment_length", type=int, default=100)
    # parser.add_argument("--sample_batch_size", type=int, default=100)
    parser.add_argument("--lr_init", type=float, default=0.0, help="Init learning rate. If 0, use default hyperparams for harvest/cleanup")
    parser.add_argument("--lr_final", type=float, default=0.0, help="Final learning rate. If 0, use default hyperparams for harvest/cleanup")
    parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Entropy coefficient (exploration/random action incentive). If 0, use default hyperparams for harvest/cleanup")

    args = parser.parse_args()

    main(args)
