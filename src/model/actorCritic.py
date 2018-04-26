import keras
import numpy
import math
import tensorflow as tf
import importlib
from keras.layers import Dense, LSTM, Softmax
from keras.models import Sequential
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model
from keras import backend as K

def relu_max(x):
    return K.relu(x, max_value=1)

class ValueNetwork(object):
    def __init__(self, parameters, modelName=None):
        self.parameters = parameters
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN

        self.gpus = self.parameters.GPUS

        self.learningRate = self.parameters.ALPHA_POLICY
        self.optimizer = self.parameters.OPTIMIZER_POLICY
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT

        self.hiddenLayer1 = self.parameters.HIDDEN_LAYER_1
        self.hiddenLayer2 = self.parameters.HIDDEN_LAYER_2
        self.hiddenLayer3 = self.parameters.HIDDEN_LAYER_3

        num_outputs = 1  # value for state

        if modelName is not None:
            self.load(modelName)
            return

        weight_initializer_range = math.sqrt(6 / (self.stateReprLen + num_outputs))
        initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                       maxval=weight_initializer_range, seed=None)
        if self.gpus > 1:
            with tf.device("/cpu:0"):
                self.model = Sequential()
                self.model.add(
                    Dense(self.hiddenLayer1, input_dim=self.stateReprLen, activation=self.activationFuncHidden,
                          bias_initializer=initializer, kernel_initializer=initializer))
                if self.hiddenLayer2 > 0:
                    self.model.add(
                        Dense(self.hiddenLayer2, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                if self.hiddenLayer3 > 0:
                    self.model.add(
                        Dense(self.hiddenLayer3, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                self.model.add(
                    Dense(num_outputs, activation='linear', bias_initializer=initializer
                          , kernel_initializer=initializer))
                self.model = multi_gpu_model(self.model, gpus=self.gpus)
        else:
            self.model = Sequential()
            hidden1 = None
            if self.parameters.NEURON_TYPE == "MLP":
                hidden1 = Dense(self.hiddenLayer1, input_dim=self.stateReprLen,
                                activation=self.activationFuncHidden,
                                bias_initializer=initializer, kernel_initializer=initializer)
            elif self.parameters.NEURON_TYPE == "LSTM":
                hidden1 = LSTM(self.hiddenLayer1, input_shape=(self.stateReprLen, 1),
                               activation=self.activationFuncHidden,
                               bias_initializer=initializer, kernel_initializer=initializer)

            self.model.add(hidden1)
            # self.valueNetwork.add(Dropout(0.5))
            hidden2 = None
            if self.hiddenLayer2 > 0:
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden2 = Dense(self.hiddenLayer2, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden2 = LSTM(self.hiddenLayer2, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.model.add(hidden2)
                # self.valueNetwork.add(Dropout(0.5))

            if self.hiddenLayer3 > 0:
                hidden3 = None
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden3 = Dense(self.hiddenLayer3, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden3 = LSTM(self.hiddenLayer3, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.model.add(hidden3)
                # self.valueNetwork.add(Dropout(0.5))

            self.model.add(
                Dense(num_outputs, activation='linear', bias_initializer=initializer
                      , kernel_initializer=initializer))

        optimizer = keras.optimizers.Adam(lr=self.learningRate)

        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.model.compile(loss='mse', optimizer=optimizer)
        self.target_model.compile(loss='mse', optimizer=optimizer)



    def load(self, modelName = None):
        if modelName is not None:
            path = "savedModels/" + modelName
            packageName = "savedModels." + modelName
            self.parameters = importlib.import_module('.networkParameters', package=packageName)
            self.loadedModelName = modelName
            self.model = load_model(path + "/value_model.h5")
            self.target_model = load_model(path + "/value_model.h5")

    def predict(self, state):
        return self.model.predict(state)[0]

    def predict_target_model(self, state):
        return self.target_model.predict(state)[0]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def save(self, path):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save(path + "value_model.h5")

class PolicyNetwork(object):
    def __init__(self, parameters, discrete, modelName = None):
        self.parameters = parameters
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN

        self.gpus = self.parameters.GPUS

        self.learningRate = self.parameters.ALPHA_POLICY
        self.optimizer = self.parameters.OPTIMIZER_POLICY
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN_POLICY
        self.hiddenLayer1 = self.parameters.HIDDEN_LAYER_1_POLICY
        self.hiddenLayer2 = self.parameters.HIDDEN_LAYER_2_POLICY
        self.hiddenLayer3 = self.parameters.HIDDEN_LAYER_3_POLICY

        self.discrete = discrete

        if discrete:
            self.actions = [[x, y, split, eject] for x in [0, 0.5, 1] for y in [0, 0.5, 1] for split in [0, 1] for
                            eject in [0, 1]]
            # Filter out actions that do a split and eject at the same time
            # Filter eject and split actions for now
            for action in self.actions[:]:
                if action[2] or action[3]:
                    self.actions.remove(action)
            self.num_actions = len(self.actions)
            self.num_outputs = self.num_actions
        else:
            self.num_outputs = 4 # x, y, split, eject all continuous between 0 and 1

        if modelName is not None:
            self.load(modelName)
            return

        weight_initializer_range = math.sqrt(6 / (self.stateReprLen + self.num_outputs))
        initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                       maxval=weight_initializer_range, seed=None)
        if self.gpus > 1:
            with tf.device("/cpu:0"):
                self.model = Sequential()
                self.model.add(
                    Dense(self.hiddenLayer1, input_dim=self.stateReprLen, activation=self.activationFuncHidden,
                          bias_initializer=initializer, kernel_initializer=initializer))
                if self.hiddenLayer2 > 0:
                    self.model.add(
                        Dense(self.hiddenLayer2, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                if self.hiddenLayer3 > 0:
                    self.model.add(
                        Dense(self.hiddenLayer3, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                self.model.add(
                    Dense(num_outputs, activation=relu_max, bias_initializer=initializer
                          , kernel_initializer=initializer))
                self.model = multi_gpu_model(self.model, gpus=self.gpus)
        else:
            self.model = Sequential()
            hidden1 = None
            if self.parameters.NEURON_TYPE == "MLP":
                hidden1 = Dense(self.hiddenLayer1, input_dim=self.stateReprLen, activation=self.activationFuncHidden,
                                bias_initializer=initializer, kernel_initializer=initializer)
            elif self.parameters.NEURON_TYPE == "LSTM":
                hidden1 = LSTM(self.hiddenLayer1, input_shape=(self.stateReprLen, 1),
                               activation=self.activationFuncHidden,
                               bias_initializer=initializer, kernel_initializer=initializer)

            self.model.add(hidden1)
            # self.valueNetwork.add(Dropout(0.5))
            hidden2 = None
            if self.hiddenLayer2 > 0:
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden2 = Dense(self.hiddenLayer2, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden2 = LSTM(self.hiddenLayer2, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.model.add(hidden2)
                # self.valueNetwork.add(Dropout(0.5))

            if self.hiddenLayer3 > 0:
                hidden3 = None
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden3 = Dense(self.hiddenLayer3, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden3 = LSTM(self.hiddenLayer3, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.model.add(hidden3)
                # self.valueNetwork.add(Dropout(0.5))
            if discrete:
                self.model.add(
                    Dense(self.num_outputs, activation="softmax", bias_initializer=initializer
                          , kernel_initializer=initializer))
            else:
                self.model.add(
                    Dense(self.num_outputs, activation=relu_max, bias_initializer=initializer
                          , kernel_initializer=initializer))

        optimizer = keras.optimizers.Adam(lr=self.learningRate)
        self.model.compile(loss='mse', optimizer=optimizer)


    def load(self, modelName = None):
        if modelName is not None:
            path = "savedModels/" + modelName
            packageName = "savedModels." + modelName
            self.parameters = importlib.import_module('.networkParameters', package=packageName)
            self.loadedModelName = modelName
            self.model = load_model(path + "/actor_model.h5")

    def predict(self, state):
        return self.model.predict(state)[0]

    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def save(self, path):
        self.model.save(path + "actor"+ "_model.h5")

    def getTarget(self, action, state):
        if self.discrete:
            target = numpy.zeros(self.num_actions)
            #target = self.model.predict(state)[0]
            target[action] = 1
        else:
            target =  action
        return numpy.array([target])

    def getAction(self, action_idx):
        return self.actions[action_idx]


class ActorCritic(object):
    def __repr__(self):
        return "AC"

    def __init__(self, parameters, num_bots, discrete, modelName = None):
        self.num_bots = num_bots
        self.actor = PolicyNetwork(parameters, discrete, modelName)
        self.critic = ValueNetwork(parameters, modelName)
        self.parameters = parameters
        self.parameters.std_dev = self.parameters.EPSILON
        self.discrete = discrete
        self.steps = 0
        self.input_len = parameters.STATE_REPR_LEN
        # Bookkeeping:
        self.lastTDE = None
        self.qValues = []

    def adjust_std_dev(self):
        if self.steps < self.parameters.STEPS_TO_MIN_NOISE:
            self.parameters.std_dev = 1 - (1 - self.parameters.MINIMUM_NOISE) *\
                                      (self.steps / self.parameters.STEPS_TO_MIN_NOISE)
        else:
            self.parameters.std_dev = self.parameters.MINIMUM_NOISE

    def updateCriticNetworks(self):
        if self.steps % self.parameters.TARGET_NETWORK_MAX_STEPS == 0:
            self.critic.update_target_model()

    def learn(self, batch):
        self.steps += 1 * self.parameters.FRAME_SKIP_RATE
        self.adjust_std_dev()
        self.train_CACLA(batch)
        self.updateCriticNetworks()

    def decideMove(self, state):
        actions = self.actor.predict(state)
        std_dev = self.parameters.std_dev
        apply_normal_dist =  [numpy.random.normal(output, std_dev) for output in actions]
        clipped = numpy.clip(apply_normal_dist, 0, 1)
        if self.discrete:
            action_idx = numpy.argmax(clipped)
            action = self.actor.getAction(action_idx)
        else:
            action_idx = None
            action =  clipped
        return action_idx, action

    def calculateTargetAndTDE(self, old_s, r, new_s, alive):
        old_state_value = self.critic.predict(old_s)
        target = r
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            updated_prediction = self.critic.predict_target_model(new_s)
            target += self.parameters.DISCOUNT * updated_prediction
        td_error = target - old_state_value
        return target, td_error

    def train_critic_CACLA(self, batch):
        len_batch = len(batch)
        inputs_critic = numpy.zeros((len_batch, self.input_len))
        targets_critic = numpy.zeros((len_batch, 1))

        # Calculate input and target for critic
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s = sample
            alive = new_s is not None
            target, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive)
            inputs_critic[sample_idx] = old_s
            targets_critic[sample_idx] = target
        old_s, a, r, new_s =  batch[-1]
        alive = new_s is not None
        target, self.lastTDE = self.calculateTargetAndTDE(old_s, r, new_s, alive)
        self.qValues.append(target)
        self.critic.train(inputs_critic, targets_critic)

    def train_actor_CACLA(self, currentExp):
        old_s, a, r, new_s = currentExp
        _, td_e = self.calculateTargetAndTDE(old_s, r, new_s, new_s is not None)
        if td_e > 0:
            input_actor = old_s
            target_actor = self.actor.getTarget(a, old_s)
            self.actor.train(input_actor, target_actor)

    def train_actor_batch_CACLA(self, batch):
        len_batch = len(batch)
        len_output = self.actor.num_outputs
        inputs = numpy.zeros((len_batch, self.input_len))
        targets = numpy.zeros((len_batch, len_output))

        # Calculate input and target for actor
        count = 0
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s = sample
            a = numpy.array([a])
            alive = new_s is not None
            _, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive)
            if td_e > 0:
                inputs[sample_idx] = old_s
                targets[sample_idx] = self.actor.getTarget(a, old_s)
                count += 1
        if count > 0:
            inputs = inputs[0:count]
            targets = targets[0:count]
            self.actor.train(inputs, targets)

    def train_CACLA(self, batch):
        # TODO: actor should not be included in the replays.. I think??? Marco says this could be done
        self.train_critic_CACLA(batch)
        #currentExp = batch[-1]
        #self.train_actor_CACLA(currentExp)
        self.train_actor_batch_CACLA(batch)



    def train(self, batch):
        inputs_critic = []
        targets_critic = []
        total_weight_changes_actor = 0
        for sample in batch:
            # Calculate input and target for critic
            old_s, a, r, new_s = sample
            alive = new_s is not None
            old_state_value = self.critic.predict(old_s)
            target = r
            if alive:
                # The target is the reward plus the discounted prediction of the value network
                updated_prediction = self.critic.predict(new_s)
                target += self.parameters.discount * updated_prediction
            td_error = target - old_state_value
            inputs_critic.append(old_s)
            targets_critic.append(target)

            # Calculate weight change of actor:
            std_dev = self.parameters.std_dev
            actor_action = self.actor.predict(old_s)
            gradient_of_log_prob_target_actor = actor_action + (a - actor_action) / (std_dev * std_dev)
            gradient = self.actor.getGradient(old_s, gradient_of_log_prob_target_actor)
            single_weight_change_actor = gradient * td_error
            total_weight_changes_actor += single_weight_change_actor

        self.critic.train(inputs_critic, targets_critic)
        self.actor.train(total_weight_changes_actor)

    def load(self, modelName):
        if modelName is not None:
            path = "savedModels/" + modelName
            packageName = "savedModels." + modelName
            self.parameters = importlib.import_module('.networkParameters', package=packageName)
            self.critic.load(path + "/critic_model.h5")
            self.actor.load(path + "/actor_model.h5")

    def save(self, path):
        self.actor.save(path)
        self.critic.save(path)

    def reset(self):
        pass

    def getTDError(self):
        return self.lastTDE

    def getQValues(self):
        return self.qValues

