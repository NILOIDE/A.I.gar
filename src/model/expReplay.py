import numpy

class ExpReplay:
    # TODO: extend with prioritized replay based on td_error. Make new specialized functions for this
    def __init__(self, parameters):
        self.memories = []
        self.max = parameters.MEMORY_CAPACITY
        self.batch_size = parameters.MEMORY_BATCH_LEN
        self.parameters = parameters

    def remember(self, new_exp):
        if len(self.memories) >= self.max:
            del self.memories[0]
        self.memories.append(new_exp)

    def rememberBatch(self, new_exps):
        difference = len(self.memories) + len(new_exps) - self.max
        if difference > 0:
            del self.memories[0:difference]
        self.memories.extend(new_exps)

    def canReplay(self):
        return len(self.memories) >= self.batch_size

    def generateTrace(self, experience):
        pass

    def sample(self):
        if self.parameters.NEURON_TYPE == "MLP":
            randIdxs = numpy.random.randint(0, len(self.memories), self.batch_size)
            return [self.memories[idx] for idx in randIdxs]
        elif self.parameters.NEURON_TYPE == "LSTM":
            trace_len = self.parameters.MEMORY_TRACE_LEN
            sampled_start_points = numpy.random.randint(0, len(self.memories) - trace_len, self.batch_size)
            sampled_traces = []
            for start_point in sampled_start_points:
                # Check that the trace does not contain any states in which the bot is dead and no states which are directly before a reset
                valid_start = False
                while not valid_start:
                    valid_start = True
                    for candidate in range(start_point, start_point + trace_len - 1):
                        if self.memories[candidate][-2] is None or self.memories[candidate][-1] == True:
                            valid_start = False
                    if not valid_start:
                        start_point = numpy.random.randint(0, len(self.memories) - trace_len)
                sampled_traces.append(self.memories[start_point:start_point + trace_len])
            return sampled_traces

    def getMemories(self):
        return self.memories
