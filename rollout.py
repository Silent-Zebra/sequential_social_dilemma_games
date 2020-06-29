"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import utility_funcs
import numpy as np
import os
import sys
import shutil
# import tensorflow as tf

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv

from DQN import DQNAgent, NeuralNet, ConvFC

import torch


# FLAGS = tf.app.flags.FLAGS
#
# tf.app.flags.DEFINE_string(
#     'vid_path', os.path.abspath(os.path.join(os.path.dirname(__file__), './videos')),
#     'Path to directory where videos are saved.')
# tf.app.flags.DEFINE_string(
#     'env', 'cleanup',
#     'Name of the environment to rollout. Can be cleanup or harvest.')
# tf.app.flags.DEFINE_string(
#     'render_type', 'pretty',
#     'Can be pretty or fast. Implications obvious.')
# tf.app.flags.DEFINE_integer(
#     'fps', 8,
#     'Number of frames per second.')


def reshape_obs_for_convfc(obs_agent_i):
    return obs_agent_i.reshape(
        obs_agent_i.shape[2], obs_agent_i.shape[0], obs_agent_i.shape[1])





class Controller(object):

    def __init__(self, env_name='harvest', num_agents=5):
        self.env_name = env_name
        if env_name == 'harvest':
            print('Initializing Harvest environment')
            self.env = HarvestEnv(num_agents=num_agents, render=True)
        elif env_name == 'cleanup':
            print('Initializing Cleanup environment')
            self.env = CleanupEnv(num_agents=num_agents, render=True)
        else:
            print('Error! Not a valid environment type')
            return

        self.num_agents = num_agents

        self.agent_policies = []
        self.agents = list(self.env.agents.values())
        # print(agents[0].action_space)
        self.action_dim = self.agents[0].action_space.n
        for _ in range(num_agents):
            # TODO right now only using 1 frame, update later to look back x (e.g. 4) frames. Later RNN/LSTM
            neural_net = ConvFC(conv_in_channels=3, # harvest specific input is 15x15x3 (HARVEST_VIEW_SIZE = 7)
                                conv_out_channels=3,
                                input_size=15,
                                hidden_size=64,
                                output_size=self.action_dim)
            self.agent_policies.append(DQNAgent(0, self.action_dim - 1, neural_net))

        self.env.reset()

    def train_agent(self, id, i, obs, action_dict, rew, next_obs, dones):
        # print(id)
        # print(i)
        agent_i = "agent-{}".format(i)
        self.agent_policies[i].q_learn_update(
            reshape_obs_for_convfc(obs[agent_i]), action_dict[agent_i],
            rew[agent_i], reshape_obs_for_convfc(next_obs[agent_i]),
            dones[agent_i])

    # def train_parallel_agents(self, id, obs, action_dict, rew, next_obs, dones):
    #     for i in range(self.num_agents):
    #         # torch.multiprocessing.spawn(self.train_agent, args=(i, obs, action_dict, rew, next_obs, dones))
    #         self.train_agent(id, i, obs, action_dict, rew, next_obs, dones)

    def rollout(self, horizon=50, save_path=None, train_agents=True):
        """ Rollout several timesteps of an episode of the environment.

        Args:
            horizon: The number of timesteps to roll out.
            save_path: If provided, will save each frame to disk at this
                location.
        """


        rewards = np.zeros(self.num_agents)
        observations = []
        shape = self.env.world_map.shape
        full_obs = [np.zeros(
            (shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]

        init_obs = self.env.reset()
        # print(init_obs)
        obs = init_obs

        for time_step in range(horizon):
            action_dim = self.action_dim
            # TODO do the DQN agent training loop now (again borrow from IPD env - should be quick and easy)
            # rand actions for all agents right now. Replace this with a train loop
            # I can use my DQN as a starting point. Later can integrate with all their stuff.
            # Start small and simple.
            # Start with just 1 past frame. Then can make it 3 or 4 past frames after
            # And then eventually use a RNN/LSTM set instead.
            action_dict = {}
            for i in range(self.num_agents):
                agent_i = "agent-{}".format(i)
                # print(obs[agent_i].shape)
                if train_agents:
                    action_dict[agent_i] = self.agent_policies[i].act(reshape_obs_for_convfc(obs[agent_i]), epsilon=self.agent_policies[i].epsilon)
                else:
                    action_dict[agent_i] = self.agent_policies[i].act(reshape_obs_for_convfc(obs[agent_i]))
                    # self.agent_policies[i].act(obs[agent_i].reshape(
                    # 1, obs[agent_i].shape[2], obs[agent_i].shape[0], obs[agent_i].shape[1] )) # batch size = 1 for 1 obs right now...


            # print(action_dict)
            # rand_action = np.random.randint(action_dim, size=self.num_agents)
            # obs, rew, dones, info, = self.env.step({'agent-0': rand_action[0],
            #                                         'agent-1': rand_action[1],
            #                                         'agent-2': rand_action[2],
            #                                         'agent-3': rand_action[3],
            #                                         'agent-4': rand_action[4]})

            next_obs, rew, dones, info, = self.env.step(action_dict)
            # print(next_obs["agent-0"].shape)
            # print(action_dict["agent-0"])
            # print(rew["agent-0"])
            # print(dones["agent-0"])

            if train_agents:
                # torch.multiprocessing.spawn(self.train_parallel_agents, nprocs=10, args=(obs, action_dict, rew, next_obs, dones))
                for i in range(self.num_agents):
                    self.train_agent(0, i, obs, action_dict, rew, next_obs, dones)

            obs = next_obs

            sys.stdout.flush()

            if save_path is not None:
                self.env.render(filename=save_path + 'frame' + str(time_step).zfill(6) + '.png')

            rgb_arr = self.env.map_to_colors()
            full_obs[time_step] = rgb_arr.astype(np.uint8)

            # rewards.append(rew)
            observations.append(obs)
            for i in range(self.num_agents):
                agent_i = "agent-{}".format(i)
                rewards[i] += rew[agent_i]
            # observations.append(obs['agent-0'])
            # rewards.append(rew['agent-0'])


        return rewards, observations, full_obs

    def render_rollout(self, horizon=50, path=None,
                       render_type='pretty', fps=8):
        """ Render a rollout into a video.

        Args:
            horizon: The number of timesteps to roll out.
            path: Directory where the video will be saved.
            render_type: Can be 'pretty' or 'fast'. Impliciations obvious.
            fps: Integer frames per second.
        """
        if path is None:
            path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
        video_name = self.env_name + '_trajectory'

        if render_type == 'pretty':
            image_path = os.path.join(path, 'frames/')
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            rewards, observations, full_obs = self.rollout(
                horizon=horizon, save_path=image_path)
            utility_funcs.make_video_from_image_dir(path, image_path, fps=fps,
                                                    video_name=video_name)

            # Clean up images
            shutil.rmtree(image_path)
        else:
            rewards, observations, full_obs = self.rollout(horizon=horizon, train_agents=False)
            utility_funcs.make_video_from_rgb_imgs(full_obs, path, fps=fps,
                                                   video_name=video_name)


# def main(unused_argv):
#     c = Controller(env_name=FLAGS.env)
#     c.render_rollout(path=FLAGS.vid_path, render_type=FLAGS.render_type,
#                      fps=FLAGS.fps)
#
#
# if __name__ == '__main__':
#     tf.app.run(main)
