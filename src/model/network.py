import keras
import os
import time
keras.backend.set_image_dim_ordering('tf')
import numpy

numpy.set_printoptions(threshold=numpy.nan)
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, LSTM, Softmax, Conv2D, MaxPooling2D, Flatten, Input, Dropout, BatchNormalization
from keras.models import Sequential
from keras.models import load_model, save_model
from keras.constraints import maxnorm

from .parameters import *

def is_square(apositiveint):
  x = apositiveint // 2
  seen = set([x])
  while x * x != apositiveint:
    x = (x + (apositiveint // x)) // 2
    if x in seen: return False
    seen.add(x)
  return True


def createDiscreteActionsCircle(numActions, enableSplit, enableEject):
    actions = []
    # Add standing still action:
    # Add all other actions:
    degPerAction = 360 / numActions
    for degreeCount in range(numActions):
        degree = math.radians(degreeCount * degPerAction)
        x = math.cos(degree)
        y = math.sin(degree)
        action = [x, y, 0, 0]
        actions.append(action)
        if enableSplit:
            splitAction = [x, y, 1, 0]
            actions.append(splitAction)
        if enableEject:
            ejectAction = [x, y, 0, 1]
            actions.append(ejectAction)
    return actions


def createDiscreteActionsSquare(numActions, enableSplit, enableEject):
    if not is_square(numActions):
        print("Number of Actions has to be a perfect square for this mode.")
        quit()
    actionsPerSide = int(math.sqrt(numActions))

    actions = []
    for row in range(actionsPerSide):
        for col in range(actionsPerSide):
            x = col / actionsPerSide + 1 / actionsPerSide / 2
            y = row / actionsPerSide + 1 / actionsPerSide / 2
            action = [x, y, 0, 0]
            actions.append(action)
            if enableSplit:
                splitAction = [x, y, 1, 0]
                actions.append(splitAction)
            if enableEject:
                ejectAction = [x, y, 0, 1]
                actions.append(ejectAction)
    return actions


class Network(object):
    def __init__(self, parameters, modelName = None):

        self.parameters = parameters

        self.gpus = self.parameters.NUM_GPUS

        # Q-learning
        self.discount = self.parameters.DISCOUNT
        self.epsilon = self.parameters.EPSILON
        self.frameSkipRate = self.parameters.FRAME_SKIP_RATE

        if self.parameters.GAME_NAME == "Agar.io":
            self.gridSquaresPerFov = self.parameters.GRID_SQUARES_PER_FOV

            # CNN
            if self.parameters.CNN_REPR:
                # (KernelSize, stride, filterNum)
                self.kernel_1 = self.parameters.CNN_L1
                self.kernel_2 = self.parameters.CNN_L2
                self.kernel_3 = self.parameters.CNN_L3

                if self.parameters.CNN_USE_L1:
                    self.stateReprLen = self.parameters.CNN_INPUT_DIM_1
                elif self.parameters.CNN_USE_L2:
                    self.stateReprLen = self.parameters.CNN_INPUT_DIM_2
                else:
                    self.stateReprLen = self.parameters.CNN_INPUT_DIM_3
            else:
                self.stateReprLen = self.parameters.STATE_REPR_LEN

            if parameters.SQUARE_ACTIONS:
                self.actions = createDiscreteActionsSquare(self.parameters.NUM_ACTIONS, self.parameters.ENABLE_SPLIT,
                                                           self.parameters.ENABLE_EJECT)
            else:
                self.actions = createDiscreteActionsCircle(self.parameters.NUM_ACTIONS, self.parameters.ENABLE_SPLIT,
                                                           self.parameters.ENABLE_EJECT)
            self.num_actions = len(self.actions)
        else:
            import gym
            env = gym.make(self.parameters.GAME_NAME)
            if self.parameters.CNN_REPR:
                pass
            else:
                self.stateReprLen = env.observation_space.shape[0]
            self.num_actions = env.action_space.n
            self.actions = list(range(self.num_actions))

        # ANN
        self.learningRate = self.parameters.ALPHA
        self.optimizer = self.parameters.OPTIMIZER
        if self.parameters.ACTIVATION_FUNC_HIDDEN == "elu":
            self.activationFuncHidden = "linear"  # keras.layers.ELU(alpha=eluAlpha)
        else:
            self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN
        self.activationFuncLSTM = self.parameters.ACTIVATION_FUNC_LSTM
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT

        self.layers = parameters.Q_LAYERS

        if self.parameters.USE_ACTION_AS_INPUT:
            inputDim = self.stateReprLen + 4
            outputDim = 1
        else:
            inputDim = self.stateReprLen
            outputDim = self.num_actions

        if self.parameters.EXP_REPLAY_ENABLED:
            input_shape_lstm = (self.parameters.MEMORY_TRACE_LEN, inputDim)
            stateful_training = False
            self.batch_len = self.parameters.MEMORY_BATCH_LEN

        else:
            input_shape_lstm = (1, inputDim)
            stateful_training = True
            self.batch_len = 1

        if self.parameters.INITIALIZER == "glorot_uniform":
            initializer = keras.initializers.glorot_uniform()
        elif self.parameters.INITIALIZER == "glorot_normal":
            initializer = keras.initializers.glorot_normal()
        else:
            weight_initializer_range = math.sqrt(6 / (self.stateReprLen + self.num_actions))
            initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                           maxval=weight_initializer_range, seed=None)

        # CNN
        if self.parameters.CNN_REPR:
            if self.parameters.CNN_P_REPR:
                # RGB
                if self.parameters.CNN_P_RGB:
                    channels = 3
                # GrayScale
                else:
                    channels = 1
                if self.parameters.CNN_LAST_GRID:
                    channels = channels * 2
                if self.parameters.COORDCONV:
                    channels += 2
                self.input = Input(shape=(self.stateReprLen, self.stateReprLen, channels))
                conv = self.input

                if self.parameters.CNN_USE_L1:
                    conv = Conv2D(self.kernel_1[2], kernel_size=(self.kernel_1[0], self.kernel_1[0]),
                                   strides=(self.kernel_1[1], self.kernel_1[1]), activation='relu',
                                   data_format='channels_last')(conv)

                if self.parameters.CNN_USE_L2:
                    conv = Conv2D(self.kernel_2[2], kernel_size=(self.kernel_2[0], self.kernel_2[0]),
                                   strides=(self.kernel_2[1], self.kernel_2[1]), activation='relu',
                                   data_format='channels_last')(conv)

                if self.parameters.CNN_USE_L3:
                    conv = Conv2D(self.kernel_3[2], kernel_size=(self.kernel_3[0], self.kernel_3[0]),
                                   strides=(self.kernel_3[1], self.kernel_3[1]), activation='relu',
                                   data_format='channels_last')(conv)

                self.valueNetwork = Flatten()(conv)

            # Not pixel input
            else:
                # Vision grid merging
                self.input = Input(shape=(self.parameters.NUM_OF_GRIDS, self.stateReprLen, self.stateReprLen))
                conv = self.input
                if self.parameters.CNN_USE_L1:
                    conv = Conv2D(self.kernel_1[2], kernel_size=(self.kernel_1[0], self.kernel_1[0]),
                                   strides=(self.kernel_1[1], self.kernel_1[1]), activation='relu',
                                   data_format='channels_first')(conv)

                if self.parameters.CNN_USE_L2:
                    conv = Conv2D(self.kernel_2[2], kernel_size=(self.kernel_2[0], self.kernel_2[0]),
                                   strides=(self.kernel_2[1], self.kernel_2[1]), activation='relu',
                                   data_format='channels_first')(conv)

                if self.parameters.CNN_USE_L3:
                    conv = Conv2D(self.kernel_3[2], kernel_size=(self.kernel_3[0], self.kernel_3[0]),
                                   strides=(self.kernel_3[1], self.kernel_3[1]), activation='relu',
                                   data_format='channels_first')(conv)

                self.valueNetwork = Flatten()(conv)

        # Fully connected layers
        if self.parameters.NEURON_TYPE == "MLP":
            layerIterable = iter(self.layers)

            regularizer = keras.regularizers.l2(self.parameters.Q_WEIGHT_DECAY)
            if self.parameters.DROPOUT:
                constraint = maxnorm(self.parameters.MAXNORM)
            else:
                constraint = None

            if parameters.CNN_REPR:
                previousLayer = self.input
                extraInputSize = self.parameters.EXTRA_INPUT
                if extraInputSize > 0:
                    extraInput = Input(shape=(extraInputSize,))
                    self.input = [self.input, extraInput]
                    denseInput = keras.layers.concatenate([self.valueNetwork, extraInput])
                    previousLayer = Dense(next(layerIterable), activation=self.activationFuncHidden,
                                        bias_initializer=initializer, kernel_initializer=initializer,
                                          kernel_regularizer=regularizer)(denseInput)
            else:
                self.input = Input(shape=(inputDim,))
                previousLayer = self.input

            for layer in layerIterable:
                if layer > 0:
                    if self.parameters.DROPOUT:
                        previousLayer = Dropout(self.parameters.DROPOUT)(previousLayer)
                    previousLayer = Dense(layer, activation=self.activationFuncHidden,
                                        bias_initializer=initializer, kernel_initializer=initializer,
                                          kernel_regularizer=regularizer, kernel_constraint=constraint)(previousLayer)
                    if self.parameters.ACTIVATION_FUNC_HIDDEN == "elu":
                        previousLayer = (keras.layers.ELU(alpha=self.parameters.ELU_ALPHA))(previousLayer)
                    if self.parameters.BATCHNORM:
                        previousLayer = BatchNormalization()(previousLayer)
            if self.parameters.DROPOUT:
                previousLayer = Dropout(self.parameters.DROPOUT)(previousLayer)

            output = Dense(outputDim, activation=self.activationFuncOutput, bias_initializer=initializer,
                           kernel_initializer=initializer, kernel_regularizer=regularizer,
                           kernel_constraint=constraint)(previousLayer)

            self.valueNetwork = keras.models.Model(inputs=self.input, outputs=output)

        elif self.parameters.NEURON_TYPE == "LSTM":

            # Hidden Layer 1
            # TODO: Use CNN with LSTM
            # if self.parameters.CNN_REPR:
            #     hidden1 = LSTM(self.hiddenLayer1, return_sequences=True, stateful=stateful_training, batch_size=self.batch_len)
            # else:
            #     hidden1 = LSTM(self.hiddenLayer1, input_shape=input_shape_lstm, return_sequences = True,
            #                    stateful= stateful_training, batch_size=self.batch_len)
            hidden1 = LSTM(self.hiddenLayer1, input_shape=input_shape_lstm, return_sequences=True,
                           stateful=stateful_training, batch_size=self.batch_len, bias_initializer=initializer
                           , kernel_initializer=initializer)(self.valueNetwork)
            # Hidden 2
            if self.hiddenLayer2 > 0:
                hidden2 = LSTM(self.hiddenLayer2, return_sequences=True, stateful=stateful_training,
                               batch_size=self.batch_len, bias_initializer=initializer
                               , kernel_initializer=initializer)(self.valueNetwork)
            # Hidden 3
            if self.hiddenLayer3 > 0:
                hidden3 = LSTM(self.hiddenLayer3, return_sequences=True, stateful=stateful_training,
                               batch_size=self.batch_len, bias_initializer=initializer
                               , kernel_initializer=initializer)(self.valueNetwork)
            # Output layer
            output = LSTM(outputDim, activation=self.activationFuncOutput,
                          return_sequences=True, stateful=stateful_training, batch_size=self.batch_len,
                          bias_initializer=initializer
                          , kernel_initializer=initializer)(self.valueNetwork)
            self.valueNetwork = keras.models.Model(inputs=self.input, outputs=output)


        # Create target network
        self.valueNetwork._make_predict_function()

        self.targetNetwork = keras.models.clone_model(self.valueNetwork)
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())

        if self.parameters.OPTIMIZER == "Adam":
            if self.parameters.GRADIENT_CLIP_NORM:
                optimizer = keras.optimizers.Adam(lr=self.learningRate, clipnorm=self.parameters.GRADIENT_CLIP_NORM,
                                                  amsgrad=self.parameters.AMSGRAD)
            elif self.parameters.GRADIENT_CLIP:
                optimizer = keras.optimizers.Adam(lr=self.learningRate, clipvalue=self.parameters.GRADIENT_CLIP,
                                                  amsgrad=self.parameters.AMSGRAD)

            else:
                optimizer = keras.optimizers.Adam(lr=self.learningRate, amsgrad=self.parameters.AMSGRAD)
        elif self.parameters.OPTIMIZER == "Nadam":
            optimizer = keras.optimizers.Nadam(lr=self.learningRate)
        elif self.parameters.OPTIMIZER == "Adamax":
            optimizer = keras.optimizers.Adamax(lr=self.learningRate)
        elif self.parameters.OPTIMIZER == "SGD":
            if self.parameters.NESTEROV:
                optimizer = keras.optimizers.SGD(lr=self.learningRate,momentum=self.parameters.NESTEROV, nesterov=True)
            else:
                optimizer = keras.optimizers.SGD(lr=self.learningRate)


        self.optimizer = optimizer

        self.valueNetwork.compile(loss='mse', optimizer=optimizer)
        self.targetNetwork.compile(loss='mse', optimizer=optimizer)

        self.model = self.valueNetwork

        if self.parameters.NEURON_TYPE == "LSTM":
            # We predict using only one state
            input_shape_lstm = (1, self.stateReprLen)
            self.actionNetwork = Sequential()
            hidden1 = LSTM(self.hiddenLayer1, input_shape=input_shape_lstm,
                           return_sequences=True, stateful=True, batch_size=1, bias_initializer=initializer
                           , kernel_initializer=initializer)
            self.actionNetwork.add(hidden1)

            if self.hiddenLayer2 > 0:
                hidden2 = LSTM(self.hiddenLayer2, return_sequences=True, stateful=True, batch_size=self.batch_len,
                               bias_initializer=initializer, kernel_initializer=initializer)
                self.actionNetwork.add(hidden2)
            if self.hiddenLayer3 > 0:
                hidden3 = LSTM(self.hiddenLayer3, return_sequences=True, stateful=True, batch_size=self.batch_len,
                               bias_initializer=initializer, kernel_initializer=initializer)
                self.actionNetwork.add(hidden3)
            self.actionNetwork.add(LSTM(self.num_actions, activation=self.activationFuncOutput,
                                        return_sequences=False, stateful=True, batch_size=self.batch_len,
                                        bias_initializer=initializer, kernel_initializer=initializer))
            self.actionNetwork.compile(loss='mse', optimizer=optimizer)

        # if __debug__:
        print(self.valueNetwork.summary())
        #     print("\n")

        if modelName is not None:
            self.load(modelName)
        self.targetNetwork._make_predict_function()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.graph = tf.get_default_graph()


    # Necessary for multiprocessing to warm up the network
    def dummy_prediction(self):
        if self.parameters.CNN_REPR:
            input_shape = ([self.parameters.NUM_OF_GRIDS, self.stateReprLen, self.stateReprLen])
        else:
            input_shape = (self.stateReprLen,)
        dummy_input = numpy.zeros(input_shape)
        dummy_input = numpy.array([dummy_input])
        self.predict(dummy_input)

    def reset_general(self, model):
        session = K.get_session()
        for layer in model.layers:
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if hasattr(v_arg, 'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
                    print('reinitializing layer {}.{}'.format(layer.name, v))

    def reset_weights(self):
        self.reset_general(self.valueNetwork)
        self.reset_general(self.targetNetwork)

    def reset_hidden_states(self):
        self.actionNetwork.reset_states()
        self.valueNetwork.reset_states()
        self.targetNetwork.reset_states()

    def load(self, modelName):
        path = modelName + "model.h5"
        self.valueNetwork = keras.models.load_model(path)
        self.targetNetwork = load_model(path)

    def setWeights(self, weights):
        self.valueNetwork.set_weights(weights)

    def trainOnBatch(self, inputs, targets, importance_weights):
        if self.parameters.NEURON_TYPE == "LSTM":
            if self.parameters.EXP_REPLAY_ENABLED:
                if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED:
                    return self.valueNetwork.train_on_batch(inputs, targets, sample_weight=importance_weights)
                else:
                    return self.valueNetwork.train_on_batch(inputs, targets)
            else:
                return self.valueNetwork.train_on_batch(numpy.array([numpy.array([inputs])]),
                                                        numpy.array([numpy.array([targets])]))
        else:
            if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED:
                return self.valueNetwork.train_on_batch(inputs, targets, sample_weight=importance_weights)
            else:
                # print("")
                # print(targets)
                return self.valueNetwork.train_on_batch(inputs, targets)

    def updateActionNetwork(self):
        self.actionNetwork.set_weights(self.valueNetwork.get_weights())

    def updateTargetNetwork(self):
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())
        if __debug__:
            print("Target Network updated.")

    def predict(self, state, batch_len=1):
        if self.parameters.NEURON_TYPE == "LSTM":
            if self.parameters.EXP_REPLAY_ENABLED:
                return self.valueNetwork.predict(state, batch_size=batch_len)
            else:
                return self.valueNetwork.predict(numpy.array([numpy.array([state])]))[0][0]
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])

                state = [grid, extra]
            else:
                state = numpy.array([state])
        with self.graph.as_default():
            prediction = self.valueNetwork.predict(state)[0]
            return prediction

    def predictTargetQValues(self, state):
        if self.parameters.USE_ACTION_AS_INPUT:
            return [self.predict_target_network(numpy.array([numpy.concatenate((state[0], act))]))[0] for act in self.actions]
        else:
            return self.predict_target_network(state)

    def predict_target_network(self, state, batch_len=1):
        if self.parameters.NEURON_TYPE == "LSTM":
            if self.parameters.EXP_REPLAY_ENABLED:
                return self.targetNetwork.predict(state, batch_size=batch_len)
            else:
                return self.targetNetwork.predict(numpy.array([numpy.array([state])]))[0][0]
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])

                state = [grid, extra]
            else:
                state = numpy.array([state])
            return self.targetNetwork.predict(state)[0]
        else:
            return self.targetNetwork.predict(state)[0]


    def predict_action_network(self, trace):
        return self.actionNetwork.predict(numpy.array([numpy.array([trace])]))[0]

    def predict_action(self, state):
        if self.parameters.USE_ACTION_AS_INPUT:
            return [self.predict(numpy.array([numpy.concatenate((state[0], act))]))[0] for act in self.actions]
        else:
            if self.parameters.NEURON_TYPE == "MLP":
                return self.predict(state)
            else:
                return self.predict_action_network(state)

    def saveModel(self, path, name=""):
        if not os.path.exists(path + "models/"):
            os.mkdir(path + "models/")
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())
        complete = False
        while not complete:
            try:
                self.targetNetwork.save(path + "models/" + name + "model.h5")
                complete = True
            except Exception:
                print("Error saving network. ########################")
                complete = False
                print("Trying to save again...")

    def setEpsilon(self, val):
        self.epsilon = val

    def setFrameSkipRate(self, value):
        self.frameSkipRate = value

    def getParameters(self):
        return self.parameters

    def getNumOfActions(self):
        return self.num_actions

    def getEpsilon(self):
        return self.epsilon

    def getDiscount(self):
        return self.discount

    def getFrameSkipRate(self):
        return self.frameSkipRate

    def getGridSquaresPerFov(self):
        return self.gridSquaresPerFov

    def getStateReprLen(self):
        return self.stateReprLen

    def getNumActions(self):
        return self.num_actions

    def getLearningRate(self):
        return self.learningRate

    def getActivationFuncHidden(self):
        return self.activationFuncHidden

    def getActivationFuncOutput(self):
        return self.activationFuncOutput

    def getOptimizer(self):
        return self.optimizer

    def getActions(self):
        return self.actions

    def getTargetNetwork(self):
        return self.targetNetwork

    def getValueNetwork(self):
        return self.valueNetwork
