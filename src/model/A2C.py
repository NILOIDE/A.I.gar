from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Input, Dense
from keras.models import Model
from model.actorCritic import ValueNetwork, PolicyNetwork, ActorCritic
import numpy as np


def relu_max(x):
    return K.relu(x, max_value=1)


def CNN_state_repr_len(parameters):
    if parameters.CNN_P_REPR:
        if parameters.CNN_P_RGB:
            channels = 3
        # GrayScale
        else:
            channels = 1
        if parameters.CNN_LAST_GRID:
            channels = channels * 2

        if parameters.CNN_USE_L1:
            return (parameters.CNN_INPUT_DIM_1,
                    parameters.CNN_INPUT_DIM_1, channels)
        elif parameters.CNN_USE_L2:
            return (parameters.CNN_INPUT_DIM_2,
                    parameters.CNN_INPUT_DIM_2, channels)
        else:
            return (parameters.CNN_INPUT_DIM_3,
                    parameters.CNN_INPUT_DIM_3, channels)
    else:
        channels = parameters.NUM_OF_GRIDS
        if parameters.CNN_USE_L1:
            return (channels, parameters.CNN_INPUT_DIM_1,
                    parameters.CNN_INPUT_DIM_1)
        elif parameters.CNN_USE_L2:
            return (channels, parameters.CNN_INPUT_DIM_2,
                    parameters.CNN_INPUT_DIM_2)
        else:
            return (channels, parameters.CNN_INPUT_DIM_3,
                    parameters.CNN_INPUT_DIM_3)


def CNN_num_conv_layers(parameters):
    if not parameters.CNN_REPR:
        return 0

    num_conv_layers = 0
    if parameters.CNN_USE_L1:
        num_conv_layers += 1
    if parameters.CNN_USE_L2:
        num_conv_layers += 1
    if parameters.CNN_USE_L3:
        num_conv_layers += 1
    return num_conv_layers


class A2C(object):
    def __init__(self, parameters):
        self.discount = 0 if parameters.END_DISCOUNT else parameters.DISCOUNT
        self.parameters = parameters
        self.std = self.parameters.GAUSSIAN_NOISE
        self.noise_decay_factor = self.parameters.AC_NOISE_DECAY
        self.networks = {}
        self.numCNNlayers = 0

        if self.parameters.GAME_NAME == "Agar.io":
            self.action_len = 2 + self.parameters.ENABLE_SPLIT + self.parameters.ENABLE_EJECT
            self.ornUhlPrev = np.zeros(self.action_len)
            if self.parameters.CNN_REPR:
                self.num_cnn_layers = CNN_num_conv_layers(self.parameters)
                self.input_len = CNN_state_repr_len(self.parameters)
            else:
                self.input_len = parameters.STATE_REPR_LEN
        else:
            import gym
            env = gym.make(self.parameters.GAME_NAME)
            if self.parameters.CNN_REPR:
                self.num_cnn_layers = CNN_num_conv_layers(self.parameters)
                self.input_len = CNN_state_repr_len(self.parameters)
            else:
                self.input_len = env.observation_space.shape[0]
            self.action_len = env.action_space.sample().shape[0]

        self.critic = None
        self.actor = None

    def createNetwork(self):
        networks = {}
        self.actor = PolicyNetwork(self.parameters, None)
        networks["MU(S)"] = self.actor
        self.critic = ValueNetwork(self.parameters, None)
        networks["V(S)"] = self.critic
        self.networks = networks

    def initializeNetwork(self, loadPath, networks=None):
        if networks is None or networks == {}:
            if networks is None:
                networks = {}

            self.actor = PolicyNetwork(self.parameters, loadPath)
            self.critic = ValueNetwork(self.parameters, loadPath)
            networks["V(S)"] = self.critic
        else:
            self.actor  = networks["MU(S)"]
            self.critic = networks["V(S)"]
        for network in networks:
            print(network + " summary:")
            if network == "Actor-Critic-Combo":
                networks[network].summary()
                continue
            networks[network].model.summary()
        self.networks = networks
        return networks

    def updateSharedLayers(self, updated_net, outdated_net):
        new_w = updated_net.get_conv_layer_weights()
        outdated_net.set_conv_layer_weights(new_w)

    def updateNoise(self):
        self.std *= self.noise_decay_factor
        if self.parameters.END_DISCOUNT:
            self.discount = 1 - self.parameters.DISCOUNT_INCREASE_FACTOR * (1 - self.discount)

    def updateCriticNetworks(self):
        self.critic.update_target_model()
        self.actor.update_target_model()

    def softlyUpdateNetworks(self):
        self.actor.softlyUpdateTargetModel()
        self.critic.softlyUpdateTargetModel()

    def updateNetworks(self):
        if self.parameters.SOFT_TARGET_UPDATES:
            self.softlyUpdateNetworks()
        else:
            self.updateCriticNetworks()

    def learn(self, batch, step):
        idxs, priorities = self.train_critic(batch)
        if step > self.parameters.AC_ACTOR_TRAINING_START:
            priorities = self.train_actor_batch(batch, priorities)
        self.updateNoise()
        if (step+1) % self.parameters.TARGET_NETWORK_STEPS == 0:
            self.updateNetworks()

        return idxs, priorities, updated_actions





