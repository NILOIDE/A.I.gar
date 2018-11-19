import keras
import numpy
import math
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from keras.layers import Conv2D, Flatten, Input, Dense
from keras.models import Model

def relu_max(x):
    return K.relu(x, max_value=1)


class ValueNetwork(object):
    def __init__(self, parameters, modelName, cnnLayers=None, cnnInput=None):
        self.parameters = parameters
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN
        self.learningRate = self.parameters.CACLA_CRITIC_ALPHA
        self.optimizer = self.parameters.OPTIMIZER_POLICY
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT

        self.layers = parameters.CACLA_CRITIC_LAYERS
        self.input = None

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

        regularizer = keras.regularizers.l2(self.parameters.CACLA_CRITIC_WEIGHT_DECAY)
        layerIterable = iter(self.layers)


        if self.parameters.CNN_REPR:
            self.input = cnnInput
            previousLayer = cnnLayers
            extraInputSize = self.parameters.EXTRA_INPUT
            if extraInputSize > 0:
                extraInput = Input(shape=(extraInputSize,))
                self.input = [cnnInput, extraInput]
                denseInput = keras.layers.concatenate([cnnLayers, extraInput])
                previousLayer = Dense(next(layerIterable), activation=self.activationFuncHidden,
                                      bias_initializer=initializer, kernel_initializer=initializer,
                                      kernel_regularizer=regularizer)(denseInput)
        else:
            self.input = keras.layers.Input((self.stateReprLen,))
            previousLayer = self.input

        for layer in layerIterable:
            if layer > 0:
                previousLayer = Dense(layer, activation=self.activationFuncHidden,
                                      bias_initializer=initializer, kernel_initializer=initializer,
                                      kernel_regularizer=regularizer)(previousLayer)
                if self.parameters.ACTIVATION_FUNC_HIDDEN_POLICY == "elu":
                    previousLayer = (keras.layers.ELU(alpha=self.parameters.ELU_ALPHA))(previousLayer)

        output = Dense(1, activation="linear", bias_initializer=initializer, kernel_initializer=initializer,
                       kernel_regularizer=regularizer)(previousLayer)

        self.model = Model(inputs=self.input, outputs=output)


        optimizer = keras.optimizers.Adam(lr=self.learningRate, amsgrad=self.parameters.AMSGRAD)

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
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])

                state = [grid, extra]
            else:
                state = numpy.array([state])
        return self.model.predict(state)[0][0]

    def predict_target_model(self, state):
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])

                state = [grid, extra]
            else:
                state = numpy.array([state])
        return self.target_model.predict(state)[0][0]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def softlyUpdateTargetModel(self):
        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            tau = self.parameters.DPG_TAU
        else:
            tau = self.parameters.CACLA_TAU
        targetWeights = self.target_model.get_weights()
        modelWeights = self.model.get_weights()
        newWeights = [targetWeights[idx] * (1 - tau) + modelWeights[idx] * tau for idx in range(len(modelWeights))]
        self.target_model.set_weights(newWeights)

    def train(self, inputs, targets, importance_weights):
        self.model.train_on_batch(inputs, targets, sample_weight=importance_weights)

    def save(self, path, name):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save(path + name + "value_model.h5")

    def getNetwork(self):
        return self.model


class PolicyNetwork(object):
    def __init__(self, parameters, modelName, cnnLayers=None, cnnInput=None):
        self.parameters = parameters
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN
        self.input = None


        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            self.learningRate = self.parameters.DPG_ACTOR_ALPHA
            self.layers = parameters.DPG_ACTOR_LAYERS
        else:
            self.learningRate = self.parameters.CACLA_ACTOR_ALPHA
            self.layers = parameters.CACLA_ACTOR_LAYERS

        self.optimizer = self.parameters.OPTIMIZER_POLICY
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN_POLICY


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

        layerIterable = iter(self.layers)

        if self.parameters.CNN_REPR:
            self.input = cnnInput
            previousLayer = cnnLayers
            extraInputSize = self.parameters.EXTRA_INPUT
            if extraInputSize > 0:
                extraInput = Input(shape=(extraInputSize,))
                self.input = [cnnInput, extraInput]
                denseInput = keras.layers.concatenate([cnnLayers, extraInput])
                previousLayer = Dense(next(layerIterable), activation=self.activationFuncHidden,
                                      bias_initializer=initializer, kernel_initializer=initializer)(denseInput)
        else:
            self.input = keras.layers.Input((self.stateReprLen,))
            previousLayer = self.input

        for neuronNumber in layerIterable:
            if neuronNumber > 0:
                previousLayer = Dense(neuronNumber, activation=self.activationFuncHidden, bias_initializer=initializer,
                                      kernel_initializer=initializer)(previousLayer)


        output = Dense(self.num_outputs, activation="sigmoid", bias_initializer=initializer,
                       kernel_initializer=initializer)(previousLayer)
        self.model = keras.models.Model(inputs=self.input, outputs=output)

        optimizer = keras.optimizers.Adam(lr=self.learningRate, amsgrad=self.parameters.AMSGRAD)

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
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])

                state = [grid, extra]
            else:
                state = numpy.array([state])
        return self.model.predict(state)[0]

    def predict_target_model(self, state):
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])

                state = [grid, extra]
            else:
                state = numpy.array([state])
        return self.target_model.predict(state)


    def train(self, inputs, targets, weights = None):
        if self.parameters.ACTOR_IS and weights is not None:
            self.model.train_on_batch(inputs, targets, sample_weight=weights)
        else:
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

    def getNetwork(self):
        return self.model
