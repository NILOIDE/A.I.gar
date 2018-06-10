import keras
import numpy
import math
import tensorflow as tf
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

        if self.parameters.INITIALIZER == "glorot_uniform":
            initializer = keras.initializers.glorot_uniform()
        elif self.parameters.INITIALIZER == "glorot_normal":
            initializer = keras.initializers.glorot_normal()
        else:
            weight_initializer_range = math.sqrt(6 / (self.stateReprLen + 1))
            initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                           maxval=weight_initializer_range, seed=None)


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

        self.model.add(Dense(num_outputs, activation='linear', bias_initializer=initializer, kernel_initializer=initializer))

        optimizer = keras.optimizers.Adam(lr=self.learningRate)

        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.model.compile(loss='mse', optimizer=optimizer)
        self.target_model.compile(loss='mse', optimizer=optimizer)

    def load(self, modelName=None):
        if modelName is not None:
            path = modelName
            self.loadedModelName = modelName
            self.model = load_model(path + "value_model.h5")
            self.target_model = load_model(path + "value_model.h5")

    def predict(self, state):
        return self.model.predict(state)[0][0]

    def predict_target_model(self, state):
        return self.target_model.predict(state)[0][0]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def softlyUpdateTargetModel(self):
        tau = self.parameters.DPG_TAU
        targetWeights = self.target_model.get_weights()
        modelWeights = self.model.get_weights()
        newWeights = [targetWeights[idx] * (1 - tau) + modelWeights[idx] * tau for idx in range(len(modelWeights))]
        self.target_model.set_weights(newWeights)

    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def save(self, path, name):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save(path + name + "value_model.h5")


class PolicyNetwork(object):
    def __init__(self, parameters, modelName=None):
        self.parameters = parameters
        self.loadedModelName = None


        if self.parameters.POLICY_OUTPUT_ACTIVATION_FUNC == "relu_max":
            policyOutputActivationFunction = relu_max
        elif self.parameters.POLICY_OUTPUT_ACTIVATION_FUNC == "sigmoid":
            policyOutputActivationFunction = "sigmoid"
        else:
            policyOutputActivationFunction = "sigmoid"

        self.stateReprLen = self.parameters.STATE_REPR_LEN

        self.gpus = self.parameters.GPUS


        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            self.learningRate = self.parameters.DPG_ACTOR_ALPHA
        else:
            self.learningRate = self.parameters.ALPHA_POLICY

        self.optimizer = self.parameters.OPTIMIZER_POLICY
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN_POLICY
        self.hiddenLayer1 = self.parameters.HIDDEN_LAYER_1_POLICY
        self.hiddenLayer2 = self.parameters.HIDDEN_LAYER_2_POLICY
        self.hiddenLayer3 = self.parameters.HIDDEN_LAYER_3_POLICY

        self.num_outputs = 2  #x, y, split, eject all continuous between 0 and 1
        if self.parameters.ENABLE_SPLIT:
            self.num_outputs += 1
        if self.parameters.ENABLE_EJECT:
            self.num_outputs += 1

        if modelName is not None:
            self.load(modelName)
            return

        if self.parameters.INITIALIZER == "glorot_uniform":
            initializer = keras.initializers.glorot_uniform()
        elif self.parameters.INITIALIZER == "glorot_normal":
            initializer = keras.initializers.glorot_normal()
        else:
            weight_initializer_range = math.sqrt(6 / (self.stateReprLen + 1))
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
                    Dense(self.num_outputs, activation=relu_max, bias_initializer=initializer
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
        self.model.add(Dense(self.num_outputs, activation=policyOutputActivationFunction, bias_initializer=initializer
                      , kernel_initializer=initializer))

        optimizer = keras.optimizers.Adam(lr=self.learningRate)
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.model.compile(loss='mse', optimizer=optimizer)
        self.target_model.compile(loss='mse', optimizer=optimizer)

    def load(self, modelName=None):
        if modelName is not None:
            path = modelName
            self.loadedModelName = modelName
            self.model = load_model(path + "actor_model.h5")

    def predict(self, state):
        return self.model.predict(state)[0]

    def predict_target_model(self, state):
        return self.target_model.predict(state)[0]


    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def update_target_model(self):
        if self.parameters.ACTOR_CRITIC_TYPE != "DPG":
            return
        self.target_model.set_weights(self.model.get_weights())

    def softlyUpdateTargetModel(self):
        if self.parameters.ACTOR_CRITIC_TYPE != "DPG":
            return
        tau = self.parameters.DPG_TAU
        targetWeights = self.target_model.get_weights()
        modelWeights = self.model.get_weights()
        newWeights = [targetWeights[idx] * (1 - tau) + modelWeights[idx] * tau for idx in range(len(modelWeights))]
        self.target_model.set_weights(newWeights)

    def save(self, path, name = ""):
        self.model.save(path + name + "actor" + "_model.h5")


class ActionValueNetwork(object):
    def __init__(self, parameters, modelName=None):
        self.parameters = parameters
        self.loadedModelName = None
        self.stateReprLen = self.parameters.STATE_REPR_LEN
        self.learningRate = self.parameters.DPG_CRITIC_ALPHA
        self.optimizer = self.parameters.OPTIMIZER
        self.activationFuncHidden = self.parameters.DPG_CRITIC_FUNC
        layers = self.parameters.DPG_CRITIC_LAYERS

        self.num_actions_inputs = 2  # x, y, split, eject all continuous between 0 and 1
        if self.parameters.ENABLE_SPLIT:
            self.num_actions_inputs += 1
        if self.parameters.ENABLE_EJECT:
            self.num_actions_inputs += 1

        if modelName is not None:
            self.load(modelName)
            return

        initializer = keras.initializers.glorot_uniform()
        input = keras.layers.Input((self.stateReprLen + self.num_actions_inputs,))
        previousLayer = input
        for neuronNumber in layers:
            previousLayer = Dense(neuronNumber, activation=self.activationFuncHidden, bias_initializer=initializer,
                          kernel_initializer=initializer)(previousLayer)

        output = Dense(1, activation="linear", bias_initializer=initializer, kernel_initializer=initializer)(previousLayer)
        self.model = keras.models.Model(inputs=input, outputs=output)


        optimizer = keras.optimizers.Adam(lr=self.learningRate)

        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.model.compile(loss='mse', optimizer=optimizer)
        self.target_model.compile(loss='mse', optimizer=optimizer)

    def load(self, modelName=None):
        if modelName is not None:
            path = modelName
            self.loadedModelName = modelName
            self.model = load_model(path + "actionValue_model.h5")
            self.target_model = load_model(path + "actionValue_model.h5")

    def predict(self, state, action):
        return self.model.predict(numpy.array([numpy.concatenate((state[0], action))]))[0][0]

    def predict_target_model(self, state, action):
        return self.target_model.predict(numpy.array([numpy.concatenate((state[0], action))]))[0][0]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def softlyUpdateTargetModel(self):
        tau = self.parameters.DPG_TAU
        targetWeights = self.target_model.get_weights()
        modelWeights = self.model.get_weights()
        newWeights = [targetWeights[idx] * (1 - tau) + modelWeights[idx] * tau for idx in range(len(modelWeights))]
        self.target_model.set_weights(newWeights)

    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def save(self, path, name):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save(path + name + "actionValue_model.h5")


class ActorCritic(object):
    def __repr__(self):
        return "AC"

    def __init__(self, parameters):
        self.discrete = False
        self.acType = parameters.ACTOR_CRITIC_TYPE
        self.parameters = parameters
        self.std = self.parameters.GAUSSIAN_NOISE
        self.noise_decay_factor = self.parameters.NOISE_DECAY
        self.steps = 0
        self.input_len = parameters.STATE_REPR_LEN
        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            self.input_len += 2 + self.parameters.ENABLE_SPLIT + self.parameters.ENABLE_EJECT
        # Bookkeeping:
        self.latestTDerror = None
        self.qValues = []
        self.actor = None
        self.critic = None
        self.combinedActorCritic = None

    def createCombinedActorCritic(self, actor, critic):
        for layer in critic.layers:
            layer.trainable = False
        mergeLayer = keras.layers.concatenate([actor.inputs[0], actor.outputs[0]])
        nonTrainableCritic = critic(mergeLayer)
        combinedModel = keras.models.Model(inputs=actor.inputs, outputs=nonTrainableCritic)
        combinedModel.compile(optimizer="Adam", loss="mse")
        return combinedModel

    def initializeNetwork(self, loadPath, networks=None):
        if networks is None or networks == {}:
            if networks is None:
                networks = {}
            if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
                self.actor = PolicyNetwork(self.parameters, loadPath)
                self.critic = ActionValueNetwork(self.parameters, loadPath)
                self.combinedActorCritic = self.createCombinedActorCritic(self.actor.model, self.critic.model)
                networks["MU(S)"] = self.actor
                networks["Q(S,A)"] = self.critic
            else:
                self.actor = PolicyNetwork(self.parameters, loadPath)
                self.critic = ValueNetwork(self.parameters, loadPath)
                networks["MU(S)"] = self.actor
                networks["V(S)"] = self.critic
        else:
            self.actor  = networks["MU(S)"]
            if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
                self.critic = networks["Q(S,A)"]
                self.combinedActorCritic = self.createCombinedActorCritic(self.actor.model, self.critic.model)
            else:
                self.critic = networks["V(S)"]
        for network in networks:
            networks[network].model.summary()
        return networks

    def updateNoise(self):
        self.std *= self.noise_decay_factor

    def updateCriticNetworks(self, time):
        if time % self.parameters.TARGET_NETWORK_STEPS == 0:
            self.critic.update_target_model()
            self.actor.update_target_model()

    def softlyUpdateNetworks(self):
        self.actor.softlyUpdateTargetModel()
        self.critic.softlyUpdateTargetModel()

    def updateNetworks(self, time):
        if self.parameters.SOFT_TARGET_UPDATES:
            self.softlyUpdateNetworks()
        else:
            self.updateCriticNetworks(time)


    def learn(self, batch):
        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            self.train_critic_DPG(batch)
            self.train_actor_DPG(batch)
        else:
            self.train_critic(batch)
            self.train_actor_batch(batch)

    def train_actor_DPG(self, batch):
        len_batch = len(batch)
        inputs = numpy.zeros((len_batch, self.parameters.STATE_REPR_LEN))
        targets = numpy.zeros((len_batch, 1))

        # Calculate input and target for actor
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s, _ = sample
            oldPrediction = self.combinedActorCritic.predict(old_s)[0]
            inputs[sample_idx] = old_s
            targets[sample_idx] = oldPrediction + self.parameters.DPG_Q_VAL_INCREASE
        self.combinedActorCritic.train_on_batch(inputs, targets)


    def train_actor_batch(self, batch):
        len_batch = len(batch)
        len_output = self.actor.num_outputs
        inputs = numpy.zeros((len_batch, self.input_len))
        targets = numpy.zeros((len_batch, len_output))

        # Calculate input and target for actor
        count = 0
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s, _ = sample
            alive = new_s is not None
            _, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive)
            target = self.calculateTarget_Actor(old_s, a, td_e)
            if target is not None:
                inputs[count] = old_s
                targets[count] = target
                count += 1
        if count > 0:
            inputs = inputs[:count]
            targets = targets[:count]
            if __debug__:
                if batch[-1][0] is inputs[-1]:
                    print("Target for current experience:", numpy.round(targets[-1], 2))
                print("Last predicted action:\t", numpy.round(self.actor.predict(inputs[-1]), 2))
                print("Last Target:\t", numpy.round(targets[-1], 2))
                print("Actor trained on number of samples: ", count)
            self.actor.train(inputs, targets)
            if __debug__:
                print("Predicted action after training:\t", numpy.round(self.actor.predict(inputs[-1]), 2))

    def applyNoise(self, action):
        #Gaussian Noise:
        apply_normal_dist = [numpy.random.normal(output, self.std) for output in action]
        return numpy.clip(apply_normal_dist, 0, 1)
        #TODO: add Ornstein-Uhlenbeck process noise with theta=0.15 and sigma=0.2


    def decideMove(self, state, bot):
        action = self.actor.predict(state)
        noisyAction = self.applyNoise(action)

        if __debug__:
            print("")
            if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
                print("Evaluation of current state-action Q(s,a): ", round(self.critic.predict(state, noisyAction), 2))
            else:
                print("Evaluation of current state V(s): ", round(self.critic.predict(state), 2))
            print("Current action:\t", numpy.round(noisyAction, 2))
            print("")

        return None, noisyAction

    def calculateTargetAndTDE(self, old_s, r, new_s, alive):
        old_state_value = self.critic.predict(old_s)
        target = r
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            updated_prediction = self.critic.predict_target_model(new_s)
            target += self.parameters.DISCOUNT * updated_prediction
        td_error = target - old_state_value
        return target, td_error


    def train_critic_DPG(self, batch):
        target, td_e = None, None
        len_batch = len(batch)
        inputs_critic = numpy.zeros((len_batch, self.input_len))
        targets_critic = numpy.zeros((len_batch, 1))

        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s, _ = sample
            alive = new_s is not None
            target = r
            if alive:
                target += self.parameters.DISCOUNT * self.critic.predict_target_model(new_s, self.actor.predict_target_model(new_s))
            inputs_critic[sample_idx]  = numpy.concatenate([old_s[0], a])
            targets_critic[sample_idx] = target
        self.qValues.append(target)
        self.critic.train(inputs_critic, targets_critic)


    def train_critic(self, batch):
        target, td_e = None, None
        len_batch = len(batch)
        inputs_critic = numpy.zeros((len_batch, self.input_len))
        targets_critic = numpy.zeros((len_batch, 1))
        # Calculate input and target for critic
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s, _ = sample
            alive = new_s is not None
            target, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive)
            inputs_critic[sample_idx] = old_s
            targets_critic[sample_idx] = target

        # Debug info:
        if target and td_e:
            self.qValues.append(target)
            self.latestTDerror = td_e

        # Train:
        self.critic.train(inputs_critic, targets_critic)


    def calculateTarget_Actor(self, old_s, a, td_e):
        target = None
        if self.acType == "CACLA":
            if td_e > 0:
                mu_s = self.actor.predict(old_s)
                target = mu_s + (a - mu_s)
            elif td_e < 0 and self.parameters.CACLA_UPDATE_ON_NEGATIVE_TD:
                mu_s = self.actor.predict(old_s)
                target = mu_s - (a - mu_s)
        elif self.acType == "Standard":
            mu_s = self.actor.predict(old_s)
            target = mu_s + td_e * (a - mu_s)

        return target


    def load(self, modelName):
        if modelName is not None:
            path = modelName
            self.critic.load(path)
            self.actor.load(path)

    def save(self, path, name = ""):
        self.actor.save(path, name)
        self.critic.save(path, name)


    def setNoise(self, val):
        self.std = val

    def setTemperature(self, val):
        self.temperature = val

    def getTemperature(self):
        return None

    def reset(self):
        self.latestTDerror = None

    def resetQValueList(self):
        self.qValues = []

    def getNoise(self):
        return self.std

    def getTDError(self):
        return self.latestTDerror

    def getQValues(self):
        return self.qValues
