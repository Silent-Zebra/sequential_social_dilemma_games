# Model taken from https://arxiv.org/pdf/1810.08647.pdf,
# INTRINSIC SOCIAL MOTIVATION VIA CAUSAL
# INFLUENCE IN MULTI-AGENT RL


# model is a single convolutional layer with a kernel of size 3, stride of size 1, and 6 output
# channels. This is connected to two fully connected layers of size 32 each

import tensorflow as tf

from ray.rllib.models.misc import normc_initializer, flatten
from ray.rllib.models.model import Model
import tensorflow.contrib.slim as slim


class ConvToFCNet(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):

        inputs = input_dict["obs"]

        # print("TYPE-IS")
        # print(type(inputs))
        # print(inputs)
        # # print(inputs.shape)
        # import sys
        # sys.stdout.flush()

        smoothed_rews = []
        if isinstance(inputs, list):
            smoothed_rews = inputs[1]
            inputs = inputs[0]

        # inputs = input_dict["obs"][0]

        # print(inputs)
        # print(inputs.shape)
        # sys.stdout.flush()

        hiddens = [32, 32]
        with tf.name_scope("custom_net"):
            inputs = slim.conv2d(
                inputs,
                6,
                [3, 3],
                1,
                activation_fn=tf.nn.relu,
                scope="conv")
            last_layer = flatten(inputs)

            last_layer = tf.concat([last_layer, smoothed_rews], axis=-1)

            i = 1
            for size in hiddens:
                label = "fc{}".format(i)
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=tf.nn.relu,
                    scope=label)
                i += 1
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="fc_out")


            # print(output)
            # print(output.shape)
            # sys.stdout.flush()


            # output = tf.concat([output, smoothed_rews], axis=-1)


            # print(output)
            # print(output.shape)

            # print("NEW OUTPUT")
            # print(output)
            # print(output.shape)
            #
            # print(last_layer)
            # sys.stdout.flush()

            return output, last_layer
