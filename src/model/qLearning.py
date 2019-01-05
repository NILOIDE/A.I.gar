import numpy
import heapq
import math
import time
from .network import Network
from keras import backend as K

def boltzmannDist(values, temp):
    maxVal = max(values)
    shiftedVals = [value - maxVal for value in values]
    distribution_values = [math.e ** (value / temp) for value in shiftedVals]
    distSum = numpy.sum(distribution_values)
    return [value / distSum for value in distribution_values]


class QLearn(object):
    def __repr__(self):
        return "Q-learning"

    def __init__(self, parameters):
        self.network = None
        self.temporalDifference = parameters.TD
        self.parameters = parameters
        self.latestTDerror = None
        self.qValues = []
        self.discount = 0 if parameters.END_DISCOUNT else parameters.DISCOUNT

        if self.parameters.GAME_NAME == "Agar.io":
            self.output_len = self.parameters.NUM_ACTIONS * (1 + self.parameters.ENABLE_SPLIT + self.parameters.ENABLE_EJECT)
            if self.parameters.CNN_REPR:
                if self.parameters.CNN_P_REPR:
                    if self.parameters.CNN_P_RGB:
                        channels = 3
                    # GrayScale
                    else:
                        channels = 1
                    if self.parameters.CNN_LAST_GRID:
                        channels = channels * 2

                    if self.parameters.CNN_USE_L1:
                        self.input_len = (self.parameters.CNN_INPUT_DIM_1,
                                          self.parameters.CNN_INPUT_DIM_1, channels)
                    elif self.parameters.CNN_USE_L2:
                        self.input_len = (self.parameters.CNN_INPUT_DIM_2,
                                          self.parameters.CNN_INPUT_DIM_2, channels)
                    else:
                        self.input_len = ( self.parameters.CNN_INPUT_DIM_3,
                                          self.parameters.CNN_INPUT_DIM_3, channels)
                else:
                    channels = self.parameters.NUM_OF_GRIDS
                    if self.parameters.CNN_USE_L1:
                        self.input_len = (channels, self.parameters.CNN_INPUT_DIM_1,
                                          self.parameters.CNN_INPUT_DIM_1)
                    elif self.parameters.CNN_USE_L2:
                        self.input_len = (channels, self.parameters.CNN_INPUT_DIM_2,
                                          self.parameters.CNN_INPUT_DIM_2)
                    else:
                        self.input_len = (channels, self.parameters.CNN_INPUT_DIM_3,
                                          self.parameters.CNN_INPUT_DIM_3)
                
            elif self.parameters.USE_ACTION_AS_INPUT:
                self.input_len = parameters.STATE_REPR_LEN + 4
                self.output_len = 1
            else:
                self.input_len = parameters.STATE_REPR_LEN
        else:
            import gym
            env = gym.make(self.parameters.GAME_NAME)
            if self.parameters.CNN_REPR:
                pass
            else:
                self.input_len = env.observation_space.shape[0]
            self.output_len = env.action_space.n

        self.discrete = True
        self.epsilon = parameters.EPSILON
        self.temperature = parameters.TEMPERATURE
        self.current_q_values = None

    def createNetwork(self):
        self.network = Network(self.parameters)

    def initializeNetwork(self, loadPath, networks = None):
        if networks is None or networks == {}:
            self.network = Network(self.parameters, loadPath)
            if networks is None:
                networks = {}
            networks["Q"] = self.network
        else:
            self.network = networks["Q"]
        return networks

    def load(self, modelName):
        if modelName is not None:
            self.network.load(modelName)

    def setValueNetWeights(self, weights):
        self.network.setWeights(weights)

    def reset(self):
        self.latestTDerror = None
        if self.parameters.NEURON_TYPE == "LSTM":
            self.network.reset_hidden_states()

    def reset_weights(self):
        self.network.reset_weights()

    def resetQValueList(self):
        self.qValues = []

    def testNetwork(self, bot, newState):
        self.network.setEpsilon(0)
        player = bot.getPlayer()
        return self.decideMove(newState, player)

    def calculateTargetForAction(self, newState, reward, alive):
        target = reward
        if alive:
            action_Q_values = self.network.predictTargetQValues(newState)
            newActionIdx = numpy.argmax(action_Q_values)
            target += self.discount * action_Q_values[newActionIdx]
        return target

    def calculateTarget(self, old_s, a, r, new_s, alive):
        old_q_values = self.network.predict(old_s)
        updated_action_value = self.calculateTargetForAction(new_s, r, alive)
        td_e = updated_action_value - old_q_values[a]
        old_q_values[a] = updated_action_value
        return old_q_values, td_e


    def calculateTDError_ExpRep_Lstm(self, exp):
        old_s, a, r, new_s, _ = exp
        alive = new_s is not None
        state_Q_values = self.network.predict_single_trace_LSTM(old_s, False)
        target = r
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            action_Q_values = self.network.predict_single_trace_LSTM(new_s, False)
            newActionIdx = numpy.argmax(action_Q_values)
            target += self.discount * action_Q_values[newActionIdx]
        q_value_of_action = state_Q_values[a]
        td_error = target - q_value_of_action
        return td_error

    def train(self, batch):
        batch_len = len(batch[0])
        # In the case of CNN, self.input_len has several dimensions
        if self.parameters.CNN_REPR:
            inputShape = numpy.array([batch_len] + list(self.input_len))
            inputs = numpy.zeros(inputShape)
        else:
            inputs = numpy.zeros((batch_len, self.input_len))
        targets = numpy.zeros((batch_len, self.output_len))
        idxs =  batch[6] if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED else None
        importance_weights = batch[5] if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED else numpy.zeros(batch_len)
        priorities = numpy.zeros_like(importance_weights)

        for sample_idx in range(batch_len):

            old_s, a, r, new_s = batch[0][sample_idx], batch[1][sample_idx], batch[2][sample_idx], batch[3][sample_idx]
            # No new state: dead
            if self.parameters.USE_ACTION_AS_INPUT:
                stateAction =  numpy.concatenate((old_s[0], self.network.actions[a]))  #old_s.extend(a)
                inputs[sample_idx] = stateAction
                if self.parameters.EXP_REPLAY_ENABLED:
                    alive = new_s.size > 1
                else:
                    alive = new_s is not None
                updatedValue = self.calculateTargetForAction(new_s, r, alive)
                oldValue = self.network.predict(numpy.array([numpy.concatenate((old_s[0], self.network.actions[a]))]))
                td_e = updatedValue - oldValue
                targets[sample_idx] = updatedValue
            else:
                inputs[sample_idx] = old_s
                if self.parameters.EXP_REPLAY_ENABLED:
                    alive = new_s.size > 1
                else:
                    alive = new_s is not None
                targets[sample_idx], td_e = self.calculateTarget(old_s, a, r, new_s, alive)
            priorities[sample_idx] = td_e

        self.network.trainOnBatch(inputs, targets, importance_weights)

        return idxs, priorities

    def learn(self, batch, step):
        idxs, priorities = self.train(batch)
        self.updateNoise()
        if (step+1) % self.parameters.TARGET_NETWORK_STEPS == 0:
            self.updateNetworks()
        self.latestTDerror = numpy.mean(priorities[-1])
        return idxs, priorities, None

    def updateNoise(self):
        self.epsilon *= self.parameters.NOISE_DECAY
        self.temperature *= self.parameters.TEMPERATURE_DECAY
        if self.parameters.END_DISCOUNT:
            self.discount = 1 - self.parameters.DISCOUNT_INCREASE_FACTOR * (1 - self.discount)

    def updateNetworks(self):
        self.updateTargetModel()
        #self.updateActionModel()

    def updateTargetModel(self):
        self.network.updateTargetNetwork()

    def decideExploration(self):
        if self.parameters.EXPLORATION_STRATEGY == "e-Greedy":
            if numpy.random.random(1) < self.epsilon:
                explore = True
                newActionIdx = numpy.random.randint(len(self.network.getActions()))
                self.qValues.append(float("NaN"))
                return explore, newActionIdx
        return False, None

    def decideMove(self, newState, updateNoise=True):
        # Take random action with probability 1 - epsilon
        explore, newActionIdx = self.decideExploration()
        if not explore:
            q_Values = self.network.predict_action(newState)
            self.current_q_values = q_Values
            if self.parameters.EXPLORATION_STRATEGY == "Boltzmann" and self.temperature != 0:
                q_Values = boltzmannDist(q_Values, self.temperature)
                action_value = numpy.random.choice(q_Values, p=q_Values)
                newActionIdx = numpy.argmax(q_Values == action_value)
            else:
                newActionIdx = numpy.argmax(q_Values)
            # Book keeping:
            self.qValues.append(q_Values[newActionIdx])
        else:
            self.current_q_values = None

        # if __debug__  and not explore and bot.player.getSelected():
        #     average_value = round(numpy.mean(q_Values), 1)
        #     q_value = round(q_Values[newActionIdx], 1)
        #     print("Expected Q-value: ", average_value, " Q(s,a) of current action: ", q_value)
        #     print("")
        newAction = self.network.actions[newActionIdx]

        if updateNoise:
            self.updateNoise()
        return newActionIdx, newAction

    def save(self, path, name = ""):
        self.network.saveModel(path, name)

    def getUpdatedParams(self):
        params = {}
        params["EPSILON"] = self.epsilon
        params["TEMPERATURE"] = self.temperature
        if self.parameters.END_DISCOUNT:
            params["DISCOUNT"] = self.discount
        return params

    def getNoise(self):
        return self.epsilon

    def setNoise(self, val):
        self.epsilon = val

    def setTemperature(self, val):
        self.temperature = val

    def setNetworkWeights(self, weightsDict):
        self.network.setWeights(weightsDict["Q"])

    def getTemperature(self):
        return self.temperature

    def getNetwork(self):
        return self.network

    def getNetworkWeights(self):
        return {"Q": self.network.getValueNetwork().get_weights()}

    def getQValues(self):
        return self.qValues

    def getTDError(self):
        return self.latestTDerror

    def getFrameSkipRate(self):
        return self.parameters.FRAME_SKIP_RATE

"""

    def updateActionModel(self, time):
        if self.parameters.NEURON_TYPE == "LSTM" and time % self.parameters.UPDATE_LSTM_MOVE_NETWORK == 0:
            self.network.updateActionNetwork()


    def train_LSTM_batch(self, batch):
        len_batch = len(batch)
        len_trace = self.parameters.MEMORY_TRACE_LEN
        old_states = numpy.zeros((len_batch, len_trace, self.input_len))
        new_states = numpy.zeros((len_batch, len_trace, self.input_len))

        action = numpy.zeros((len_batch, len_trace))
        reward = numpy.zeros((len_batch, len_trace))

        for i in range(len_batch):
            for j in range(len_trace):
                old_states[i, j, :] = batch[i][j][0]
                action[i, j] = batch[i][j][1]
                reward[i, j] = batch[i][j][2]
                new_states[i, j, :] = batch[i][j][3]

        #TODO: Why is len_batch needed? The shape seems to be 32, 10, 407 for some reason
        #print("Shape old_states: ", numpy.shape(old_states))
        target = self.network.predict(old_states, len_batch)

        #TODO: agent might have died so do not predict reward there with new State as new state is None then
        target_val = self.network.predict_target_network(new_states, len_batch)

        for i in range(len_batch):
            for j in range(self.parameters.TRACE_MIN, len_trace):
                a = numpy.argmax(target_val[i][j])
                # TODO: target is reward if the agent died.
                target[i][j][int(action[i][j])] = reward[i][j] + self.parameters.DISCOUNT * (target_val[i][j][a])
        #print("Weights before training:")
        #print(self.network.valueNetwork.get_weights()[0])
        self.network.trainOnBatch(old_states, target)
        #print("After training:")
        #print(self.network.valueNetwork.get_weights()[0])


#TODO: With lstm neurons we need one network python object per learning agent in the environment.
#TODO: !! important if at some point we want to train multiple lstm agents

    def train_LSTM(self, batch):
        if self.parameters.EXP_REPLAY_ENABLED:
            # The last memory is not a memory trace, therefore useless for exp replay lstm
            if len(batch) > self.parameters.MEMORY_BATCH_LEN:
                self.train_LSTM_batch(batch[:-1])
        else:
            old_s, a, r, new_s, reset = batch[-1]

            target = self.network.predict(old_s)
            # Debug: print predicted q values:
            average_value = round(numpy.mean(target), 2)
            q_value = round(target[a], 2)
            print("Expected Q-value: ", average_value, " Q(s,a) of current action: ", q_value)

            # Calculate target for action that we took:
            alive = (new_s is not None)
            target_for_action = self.calculateTargetForAction(new_s, r, alive)
            self.latestTDerror = target_for_action - target[a] # BookKeeping
            target[a] = target_for_action
            print("Reward: ", round(r, 2))
            print("Target for action: ", round(target_for_action, 2))
            print("TD-Error: ", self.latestTDerror)
            loss = self.network.trainOnBatch(old_s, target)
            print("Loss: ", loss)
            print("")

#TODO: Get loss from all trainOnBatch calls, store them in an array and plot them

"""
