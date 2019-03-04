import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #This suppresses tensorflow AVX warnings
import importlib
import importlib.util
import shutil
# import psutil
import distutils
import time
import datetime
from controller.controller import Controller
from model.qLearning import QLearn
from model.actorCritic import ActorCritic
from model.bot import *
from model.model import Model
from model.expReplay import ExpReplay
from model.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import matplotlib.pyplot as plt
from builtins import input
import multiprocessing as mp
from time import sleep
import gym

from view.view import View
from modelCombiner import createCombinedModelGraphs, plot
import hashlib

import numpy as np



def fix_seeds(seedNum):
    import tensorflow as tf

    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

    # import os
    # os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting np generated random numbers
    # in a well-defined initial state.

    if seedNum is not None:
        np.random.seed(42)
    else:
        np.random.seed()

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    # rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    if seedNum is not None:

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

        from keras import backend as K

        # The below tf.set_random_seed() will make random number generation
        # in the TensorFlow backend have a well-defined initial state.
        # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

        tf.set_random_seed(seedNum)

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
    else:
        session_conf = tf.ConfigProto()

        from keras import backend as K

        # The below tf.set_random_seed() will make random number generation
        # in the TensorFlow backend have a well-defined initial state.
        # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)



def initModelFolder(name = None, loadedModelName = None, model_in_subfolder = None):
    if name is None:
        path, startTime = createPath()
    else:
        if loadedModelName is None:
            path, startTime = createNamedPath(name)
        else:
            if model_in_subfolder:
                path, startTime = createNamedLoadPath(name, loadedModelName)
            else:
                path, startTime = createLoadPath(loadedModelName)
            distutils.dir_util.copy_tree(loadedModelName + "models/", path + "models/")
    copyParameters(path, loadedModelName)
    return path, startTime


def getPackageName(path):
    packageName = list(path[0:len(path)-1])
    for idx in range(len(packageName)):
        if packageName[idx] == "/":
            packageName[idx] = "."
    return "".join(packageName)


def createPath():
    basePath = "savedModels/"
    if not os.path.exists(basePath):
        os.makedirs(basePath)
    now = datetime.datetime.now()
    startTime = now
    nowStr = now.strftime("%b-%d_%H:%M")
    path = basePath + "$" + nowStr + "$"
    # Also display seconds in name if we already have a model this minute
    if os.path.exists(path):
        nowStr = now.strftime("%b-%d_%H:%M:%S")
        path = basePath + "$" + nowStr + "$"
    os.makedirs(path)
    path += "/"
    print("Path: ", path)
    return path, startTime

def countLoadDepth(loadedModelName):
    if loadedModelName[-3] == ")" and loadedModelName[-6:-4] == "(l":
        loadDepth = int(loadedModelName[-4]) + 1
    else:
        loadDepth = 1
    loadString = "_(l" + str(loadDepth) + ")"
    return loadString

def createLoadPath(loadedModelName):
    loadDepth = countLoadDepth(loadedModelName)
    basePath = "savedModels/"
    if not os.path.exists(basePath):
        os.makedirs(basePath)
    now = datetime.datetime.now()
    startTime = now
    nowStr = now.strftime("%b-%d_%H:%M")
    path = basePath + "$" + nowStr + loadDepth + "$"
    # Also display seconds in name if we already have a model this minute
    if os.path.exists(path):
        nowStr = now.strftime("%b-%d_%H:%M:%S")
        path = basePath + "$" + nowStr + loadDepth + "$"
    os.makedirs(path)
    path += "/"
    print("Path: ", path)
    return path, startTime

def countNamedLoadDepth(superName, loadedModelName):
    char = -3
    while loadedModelName[char] != "/":
        char -= 1
    if loadedModelName[char-1] == ")" and loadedModelName[char-4:char-2] == "(l":
        loadDepth = int(loadedModelName[char-2]) + 1
    else:
        loadDepth = 1
    loadString = "_(l" + str(loadDepth) + ")"
    superName = superName[0:len(superName)-1] + loadString + "/"
    return superName

def createNamedLoadPath(superName, loadedModelName):
    superName = countNamedLoadDepth(superName, loadedModelName)
    basePath = "savedModels/"
    if not os.path.exists(basePath):
        os.makedirs(basePath)
    # Create subFolder for given parameter tweaking
    osPath = os.getcwd() + "/" + superName
    time.sleep(np.random.rand())
    if not os.path.exists(osPath):
        os.makedirs(osPath)
    # Create folder based on name
    now = datetime.datetime.now()
    startTime = now
    nowStr = now.strftime("%b-%d_%H:%M:%S:%f")
    path = superName + "$" + nowStr + "$"
    time.sleep(np.random.rand())
    if os.path.exists(path):
        randNum = np.random.randint(100000)
        path = superName + "$" + nowStr + "-" + str(randNum) + "$"
    os.makedirs(path)
    path += "/"
    print("Super Path: ", superName)
    print("Path: ", path)
    return path, startTime

def createNamedPath(superName):
    #Create savedModels folder
    basePath = "savedModels/"
    if not os.path.exists(basePath):
        os.makedirs(basePath)
    #Create subFolder for given parameter tweaking
    osPath = os.getcwd() + "/" + superName
    time.sleep(np.random.rand())
    if not os.path.exists(osPath):
        os.makedirs(osPath)
    #Create folder based on name
    now = datetime.datetime.now()
    startTime = now
    nowStr = now.strftime("%b-%d_%H:%M:%S:%f")
    path = superName  + "$" + nowStr + "$"
    time.sleep(np.random.rand())
    if os.path.exists(path):
        randNum = np.random.randint(100000)
        path = superName + "$" + nowStr + "-" + str(randNum) + "$"
    os.makedirs(path)
    path += "/"
    print("Super Path: ", superName)
    print("Path: ", path)
    return path, startTime


def copyParameters(path, loadedModelName = None):
    # Copy the simulation, NN and RL parameters so that we can load them later on
    if loadedModelName is None:
        shutil.copy("model/networkParameters.py", path)
        shutil.copy("model/parameters.py", path)
    else:
        shutil.copy(loadedModelName + "networkParameters.py", path)
        shutil.copy(loadedModelName + "parameters.py", path)
        if os.path.exists(loadedModelName + "model.h5"):
            shutil.copy(loadedModelName + "model.h5", path)
        if os.path.exists(loadedModelName + "actor_model.h5"):
            shutil.copy(loadedModelName + "actor_model.h5", path)
        if os.path.exists(loadedModelName + "value_model.h5"):
            shutil.copy(loadedModelName + "value_model.h5", path)


def setSeedAccordingToFolderNumber(model_in_subfolder, loadModel, modelPath, enableTrainMode):
    seedNumber = None
    if model_in_subfolder and not loadModel:
        folders = [i for i in os.listdir(modelPath) if os.path.isdir(modelPath + "/" + i)]
        seedNumber = len(folders)
    if seedNumber and enableTrainMode:
        fix_seeds(seedNumber)


# TODO: Remove the need for these two functions. make it so that algorithm type has to be set from parameters
def algorithmNumberToName(val):
    if val == 0:
        return "Q-Learning"
    elif val == 2:
        return "CACLA"
    elif val == 3:
        return "Discrete ACLA"
    else:
        print("Wrong algorithm selected...")
        quit()


def algorithmNameToNumber(name):
    name = str(name)
    if name == "Q-learning":
        return 0
    elif name == "CACLA":
        return 2
    elif name == "Discrete ACLA":
        return 3
    else:
        print("ALGORITHM in networkParameters not found.\n")
        quit()


def checkValidParameter(param):
    name_of_file = "model/networkParameters.py"
    lines = open(name_of_file, 'r').readlines()
    for n in range(len(lines)):
        name = ""
        for char in lines[n]:
            if char == " ":
                break
            name += char
        if param == name:
            return n
    print("Parameter with name " + param + "not found.")
    quit()


def modifyParameterValue(tweaked, path):
    name_of_file = path + "networkParameters.py"
    lines = open(name_of_file, 'r').readlines()
    for i in range(len(tweaked)):
        text = ""
        for char in lines[tweaked[i][2]]:
            text += char
            if char == "=":
                break
        text += " " + str(tweaked[i][1]) + "\n"
        lines[tweaked[i][2]] = text
    out = open(name_of_file, 'w')
    out.writelines(lines)
    out.close()


def nameSavedModelFolder(array):
    name = ""
    for i in range(len(array)):
        if i != 0:
            name += "&"
        name += array[i][0] + "=" + str(array[i][1]).replace('.', '_')
    name += '/'
    return name


def modelMustHavePlayers():
    print("Model must have players")
    quit()


def fitsLimitations(number, limit):
    if number < 0:
        print("Number can't be negative.")
        quit()
    if number > limit:
        print("Number can't be larger than ", limit, ".")
        quit()
    return True


def defineScreenSize(humansNr):
    # Define screen size (to allow splitscreen)
    if humansNr == 2:
        return int(SCREEN_WIDTH * humansNr + humansNr - 1), int(SCREEN_HEIGHT)
    if humansNr == 3:
        return int(SCREEN_WIDTH * humansNr * 2 / 3) + humansNr - 1, int(SCREEN_HEIGHT * 2 / 3)

    return SCREEN_WIDTH, SCREEN_HEIGHT


def elapsedTimeText(secondsElapsed):
    seconds_print = int(secondsElapsed % 60)
    minutesElapsed = int(secondsElapsed / 60)
    minutes_print = minutesElapsed % 60
    hoursElapsed = int(minutesElapsed / 60)
    return str(hoursElapsed) + "h : " + str(minutes_print) + "m : " + str(seconds_print) + "s"


def printTrainProgress(parameters, currentPart, startTime):
    smallPart = max(int(parameters.MAX_TRAINING_STEPS / 100), 1)  # Get int value closest to to 1% of training time
    currentPart += smallPart
    percentPrint = currentPart / parameters.MAX_TRAINING_STEPS
    timePrint = elapsedTimeText(int(time.time()- startTime))
    totalTime = (time.time() - startTime) / currentPart * parameters.MAX_TRAINING_STEPS
    totalTimePrint = elapsedTimeText(int(totalTime))
    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n" +
          "Trained:                           " + str(int(percentPrint * 100)) + "%\n" +
          "Time elapsed:                      " + timePrint + "\n" +
          "Estimated total duration time:     " + totalTimePrint + "\n" +
          "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    return currentPart


def createNetwork(parameters, path):
    if not os.path.exists(path + "models/"):
        os.mkdir(path + "models/")
    algorithm = parameters.ALGORITHM
    learningAlg = None
    if algorithm == "Q-learning":
        learningAlg = QLearn(parameters)
    elif algorithm == "CACLA":
        learningAlg = ActorCritic(parameters)
    elif algorithm == "DPG":
        learningAlg = ActorCritic(parameters)
    elif algorithm == "SPG":
        learningAlg = ActorCritic(parameters)
    else:
        print("Wrong algorithm name in parameters.")
        quit()
    learningAlg.createNetwork()
    learningAlg.save(path)
    if parameters.ENABLE_TRAINING:
        learningAlg.save(path, "0_")


def createHumans(numberOfHumans, model1):
    for i in range(numberOfHumans):
        name = input("Player" + str(i + 1) + " name:\n")
        model1.createHuman(name)


def createBots(number, model, botType, parameters, networkLoadPath=None):
    algorithm = parameters.ALGORITHM
    learningAlg = None
    # loadPath = model.getPath() if loadModel else None
    if botType == "NN":
        Bot.num_NNbots = number
        networks = {}
        for i in range(number):
            # TODO: Since bots only use the learnAlg for moving, we could make it into a separate class (or subclass)
            # Create algorithm instance
            if algorithm == "Q-learning":
                learningAlg = QLearn(parameters)
            elif algorithm == "CACLA":
                learningAlg = ActorCritic(parameters)
            elif algorithm == "DPG":
                learningAlg = ActorCritic(parameters)
            elif algorithm == "SPG":
                learningAlg = ActorCritic(parameters)
            else:
                print("Please enter a valid algorithm.\n")
                quit()

            networks = learningAlg.initializeNetwork(networkLoadPath, networks)
            model.createBot(botType, learningAlg, parameters)
    elif botType == "Greedy":
        Bot.num_Greedybots = number
        for i in range(number):
            model.createBot(botType, None, parameters)
    elif botType == "Random":
        for i in range(number):
            model.createBot(botType, None, parameters)


def createTestParams(packageName, virus=None, num_nn_bots=1, num_greedy_bots=0, num_rand_bots=0):
    # Create a copy of the networkParameters module import without
    # overwriting the already-existing global networkParameters module
    SPEC_OS = importlib.util.find_spec('.networkParameters', package=packageName)
    testParameters = importlib.util.module_from_spec(SPEC_OS)
    SPEC_OS.loader.exec_module(testParameters)
    del SPEC_OS

    # Change parameters in testParameters module
    testParameters.GATHER_EXP = False
    testParameters.EPSILON = 0
    testParameters.TEMPERATURE = 0
    testParameters.GAUSSIAN_NOISE = 0
    if virus != None:
        testParameters.VIRUS_SPAWN = virus
    testParameters.NUM_NN_BOTS = num_nn_bots
    testParameters.NUM_GREEDY_BOTS = num_greedy_bots
    testParameters.NUM_RAND_BOTS = num_rand_bots
    return testParameters


def finalPathName(parameters, path):
    steps = parameters.MAX_TRAINING_STEPS
    path = path[:-1]
    suffix = ""
    if steps >= 1000000000:
        suffix = "B"
        steps = int(steps / 1000000000)
    elif steps >= 1000000:
        suffix = "M"
        steps = int(steps / 1000000)
    elif steps >= 1000:
        suffix = "K"
        steps = int(steps / 1000)
    updatedPath = path + "_" + str(steps) + suffix + "/"
    os.rename(path, updatedPath)
    return updatedPath

# Export test result data into .txt files
def exportResults(results, path, name):
    filePath = path + name + ".txt"
    with open(filePath, "a") as f:
        for val in results:
            line = str(val) + "\n"
            f.write(line)

# Plot test result and export plot to file with given name
def plotTesting(testResults, path, timeBetween, end):
    for testType in testResults[0]:
        x = range(0, len(testResults)*timeBetween, timeBetween)
        y = [x[testType]["meanScore"] for x in testResults]
        ysigma = [x[testType]["stdMean"] for x in testResults]

        y_lower_bound = [y[i] - ysigma[i] for i in range(len(y))]
        y_upper_bound = [y[i] + ysigma[i] for i in range(len(y))]

        plt.ticklabel_format(axis='x', style='sci', scilimits=(1, 4))
        plt.clf()
        fig = plt.gcf()
        ax = plt.gca()
        # fig, ax = plt.subplots(1)
        ax.plot(x, y, lw=2, label="testing mass", color='blue')
        ax.fill_between(x, y_lower_bound, y_upper_bound, facecolor='blue', alpha=0.5,
                        label="+/- sigma")
        ax.set_xlabel("Time")
        yLabel = "Mass"
        title =  testResults[0][testType]["plotName"] + " mass over time"

        meanY = np.mean(y)
        ax.legend(loc='upper left')
        ax.set_ylabel(yLabel)
        ax.set_title(title + " mean value (" + str(round(meanY, 1)) + ") $\pm$ $\sigma$ interval")
        ax.grid()
        fig.savefig(path + title + ".pdf")

        plt.close()


# Plot test results and export data
def exportTestResults(testResults, path, parameters):
    maxSteps = parameters.MAX_TRAINING_STEPS
    if not os.path.exists(path + "data/"):
        os.mkdir(path + "data/")
    meanMassesOfTestResults = [val["current"]["meanScore"] for val in testResults]
    exportResults(meanMassesOfTestResults, path + "data/", "testMassOverTime")
    meanMassesOfPelletResults = [val["pellet"]["meanScore"] for val in testResults]
    exportResults(meanMassesOfPelletResults, path + "data/", "Pellet_CollectionMassOverTime")

    if parameters.MULTIPLE_BOTS_PRESENT:
        meanMassesOfGreedyResults = [val["vsGreedy"]["meanScore"] for val in testResults]
        exportResults(meanMassesOfGreedyResults, path + "data/", "VS_1_GreedyMassOverTime")
    if parameters.VIRUS_SPAWN:
        meanMassesOfPelletVirusResults = [val["virus"]["meanScore"] for val in testResults]
        exportResults(meanMassesOfPelletVirusResults, path + "data/", "Pellet_Collection_Virus_MassOverTime")
        if parameters.MULTIPLE_BOTS_PRESENT:
            meanMassesOfGreedyVirusResults = [val["virusGreedy"]["meanScore"] for val in testResults]
            exportResults(meanMassesOfGreedyVirusResults, path + "data/", "VS_1_Greedy_Virus_MassOverTime")

    testInterval =int(maxSteps/100 *parameters.TRAIN_PERCENT_TEST_INTERVAL)
    plotTesting(testResults, path, testInterval, maxSteps)


# Perform 1 episode of the test. Return the mass over time list, the mean mass of the episode, and the max mass.
def performAgarioTest(testNetworkPath, specialParams):
    testParams = createTestParams(*specialParams)
    testModel = Model(False, False, testParams)
    createModelPlayers(testParams, testModel, testNetworkPath)
    testModel.initialize()
    for step in range(testParams.RESET_LIMIT):
        testModel.update()

    bots = testModel.getBots()
    massOverTime = [bot.getMassOverTime() for bot in bots]
    meanMass = np.mean([np.mean(botMass) for botMass in massOverTime])
    maxMeanMass = np.max(meanMass)
    maxMass = np.max([np.max(botMass) for botMass in massOverTime])
    varianceMass = np.mean(np.var(massOverTime))

    return [massOverTime, meanMass, maxMass, os.getpid()]


# Test the model for 'n' amount of episodes for the given type of test. This is done in parallel uses a pool of workers.
# The test results from all tests are put together into 1 structure to then be used for averaging and plotting.
def testAgarioModel(testNetworkPath, testType, plotName, specialParams, n_tests, testName):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Testing", testType, "...\n")

    start = time.time()
    # Parallel testing
    pool = mp.Pool(n_tests)
    print("Initializing " + str(n_tests) + " testers..." )
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    testResults = pool.starmap(performAgarioTest, [(testNetworkPath, specialParams) for process in range(n_tests)])
    pool.close()
    pool.join()
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(testName + " test's " + testType + " runs finished.\n")
    for result in testResults:
        print("Process id #" + str(result[3]) + "'s test run:" +
              "\n   Number of bot mass lists: " + str(len(result[0])) +
              "\n   Mean mass: " + str(result[1]) +
              "\n   Max mass: " + str(result[2]) +
              "\n")
    print("Number of tests:   " + str(n_tests) + "\n"
          "Time elapsed:      " + str.format('{0:.3f}', time.time() - start) + "s")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    masses = []
    meanMasses = []
    maxMasses = []
    for test in range(n_tests):
        masses.extend(testResults[test][0])
        meanMasses.append(testResults[test][1])
        maxMasses.append(testResults[test][2])
    evals = {"name":testType}
    evals["plotName"] = plotName
    evals["meanScore"] = np.mean(meanMasses)
    evals["stdMean"] = np.std(meanMasses)
    evals["meanMaxScore"] = np.mean(maxMasses)
    evals["stdMax"] = np.std(maxMasses)
    evals["maxScore"] = np.max(maxMasses)
    return evals, masses


def testingProcedure(path, testNetworkPath, parameters, testName, n_tests):
    # TODO: Perform all test kinds simultaneously
    testEvals = {}
    masses = {}
    packageName = getPackageName(path)
    testParams = [packageName]
    testEvals["current"], masses["current"] = testAgarioModel(testNetworkPath, "test", "Test", testParams, n_tests, testName)

    pelletTestParams = [packageName, False]
    testEvals["pellet"], masses["pellet"] = testAgarioModel(testNetworkPath, "pellet", "Pellet_Collection", pelletTestParams,
                                                      n_tests, testName)
    if parameters.MULTIPLE_BOTS_PRESENT:
        greedyTestParams = (packageName, False, 1, 1)
        testEvals["vsGreedy"], masses["vsGreedy"] = testAgarioModel(testNetworkPath, "vsGreedy", "Vs_Greedy", greedyTestParams,
                                                              n_tests, testName)
    if parameters.VIRUS_SPAWN:
        virusTestParams = (packageName, True)
        testEvals["virus"], masses["virus"] = testAgarioModel(testNetworkPath, "pellet_with_virus", "Pellet_Collection_with_Viruses",
                                                        virusTestParams, n_tests, testName)
        if parameters.MULTIPLE_BOTS_PRESENT:
            virusGreedyTestParams = (packageName, True, 1, 1)
            testEvals["virusGreedy"], masses["virusGreedy"] = testAgarioModel(testNetworkPath, "vsGreedy_with_virus", "Vs_Greedy_with_Viruses",
                                                                        virusGreedyTestParams, n_tests, testName)
    return testEvals, masses


def runFinalTests(path, parameters):
    testNetworkPath = path + "models/"
    evals, masses = testingProcedure(path, testNetworkPath, parameters, "Final", parameters.FINAL_TEST_NUM)
    # TODO: add more test scenarios for multiple greedy bots and full model check

    name_of_file = path + "/final_results.txt"
    with open(name_of_file, "w") as file:
        data = "Number of runs per testing: " + str(parameters.FINAL_TEST_NUM) + "\n"
        for testType in evals:
            name = evals[testType]["name"]
            maxScore = str(round(evals[testType]["maxScore"], 1))
            meanScore = str(round(evals[testType]["meanScore"], 1))
            stdMean = str(round(evals[testType]["stdMean"], 1))
            meanMaxScore = str(round(evals[testType]["meanMaxScore"], 1))
            stdMax = str(round(evals[testType]["stdMax"], 1))
            data += name + " Highscore: " + maxScore + " Mean: " + meanScore + " StdMean: " + stdMean \
                    + " Mean_Max_Score: " + meanMaxScore + " Std_Max_Score: " + stdMax + "\n"
        file.write(data)

    for testType in masses:
        print("\nPlotting " + evals[testType]["plotName"] + "...\n")

        meanMassPerTimeStep = []
        for timeIdx in range(parameters.RESET_LIMIT):
            val = 0
            for test in masses[testType]:
                val += test[timeIdx]
            meanVal = val / parameters.FINAL_TEST_NUM
            meanMassPerTimeStep.append(meanVal)
        # exportTestResults(meanMassPerTimeStep, modelPath, "Mean_Mass_" + name)
        labels = {"meanLabel": "Mean Mass", "sigmaLabel": '$\sigma$ range', "xLabel": "Step number",
                  "yLabel": "Mass mean value", "title": "Mass plot test phase", "path": path,
                  "subPath": "Mean_Mass_" + str(evals[testType]["plotName"])}
        plot(masses[testType], parameters.RESET_LIMIT, 1, labels)


def performGymTest(testNetworkPath, specialParams):
    testParams = createTestParams(*specialParams)
    # Create game instance
    networkLoadPath = testNetworkPath
    env = gym.make(testParams.GAME_NAME)
    learningAlg = None
    if testParams.ALGORITHM == "Q-learning":
        learningAlg = QLearn(testParams)
    elif testParams.ALGORITHM == "CACLA":
        learningAlg = ActorCritic(testParams)
    elif testParams.ALGORITHM == "DPG":
        learningAlg = ActorCritic(testParams)
    elif testParams.ALGORITHM == "SPG":
        learningAlg = ActorCritic(testParams)
    else:
        print("Please enter a valid algorithm.\n")
        quit()
    networks = {}
    networks = learningAlg.initializeNetwork(networkLoadPath, networks)
    observation = np.array([env.reset()])

    reward_list = []
    rewardSum = 0
    done = False
    # Run game until terminated
    while True:
        actionIdx, action = learningAlg.decideMove(observation)
        if testParams.ALGORITHM in {"CACLA", "DPG", "SPG"}:
            actionIdx = np.argmax(action)
        for _ in range(testParams.FRAME_SKIP_RATE):
            observation, reward, done, info = env.step(actionIdx)
            observation = np.array([observation])
            rewardSum += reward
            if done:
                if testParams.GAME_NAME[:8] == "CartPole":
                    reward_list.append(-1.0)
                break

        reward_list.append(rewardSum)
        rewardSum = 0
        if done:
            break
    return [reward_list, np.mean(reward_list), np.max(reward_list), os.getpid()]


def gymTestingProcedure(path, testNetworkPath, parameters, testName, n_tests):
    # TODO: Perform all test kinds simultaneously
    testEvals = {}
    packageName = getPackageName(path)
    testParams = [packageName]
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Testing", parameters.GAME_NAME, "...\n")

    start = time.time()
    # Parallel testing
    pool = mp.Pool(n_tests)
    print("Initializing " + str(n_tests) + " testers...")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    testResults = pool.starmap(performGymTest, [(testNetworkPath, testParams) for process in range(n_tests)])
    pool.close()
    pool.join()
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(testName + " test's runs finished.\n")
    for result in testResults:
        print("Process id #" + str(result[3]) + "'s test run:" +
              "\n   Number of rewards in test: " + str(len(result[0])) +
              "\n   Mean reward: " + str(result[1]) +
              "\n   Max reward: " + str(result[2]) +
              "\n")
    print("Number of tests:   " + str(n_tests) + "\n" +
          "Time elapsed:      " + str.format('{0:.3f}', time.time() - start) + "s")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    evals = {"name": testName}
    rewards = [reward_list[0] for reward_list in testResults]
    testResults = np.array(testResults)
    evals["meanScore"] = np.mean(testResults[:,1])
    evals["stdMean"] = np.std(testResults[:,1])
    evals["meanMaxScore"] = np.mean(testResults[:,2])
    evals["stdMax"] = np.std(testResults[:,2])
    evals["maxScore"] = np.max(testResults[:,2])

    return testEvals, rewards


def runFinalGymTests(path, parameters):
    testNetworkPath = path + "models/"
    evals, rewards = gymTestingProcedure(path, testNetworkPath, parameters, "Final", parameters.FINAL_TEST_NUM)

    name_of_file = path + "/final_results.txt"
    with open(name_of_file, "w") as file:
        data = "Number of runs per testing: " + str(parameters.FINAL_TEST_NUM) + "\n"
        for testType in evals:
            name = evals[testType]["name"]
            maxScore = str(round(evals[testType]["maxScore"], 1))
            meanScore = str(round(evals[testType]["meanScore"], 1))
            stdMean = str(round(evals[testType]["stdMean"], 1))
            meanMaxScore = str(round(evals[testType]["meanMaxScore"], 1))
            stdMax = str(round(evals[testType]["stdMax"], 1))
            data += name + " Highscore: " + maxScore + " Mean: " + meanScore + " StdMean: " + stdMean \
                    + " Mean_Max_Score: " + meanMaxScore + " Std_Max_Score: " + stdMax + "\n"
        file.write(data)

    print("\nPlotting test run...\n")
    labels = {"meanLabel": "Mean Reward", "sigmaLabel": '$\sigma$ range', "xLabel": "Step number",
              "yLabel": "Reward mean value", "title": "Reward plot test phase", "path": path,
              "subPath": "Mean_Reward"}

    plot(rewards, parameters.RESET_LIMIT, 1, labels)


def createModelPlayers(parameters, model, networkLoadPath=None, numberOfHumans=0):
    numberOfNNBots = parameters.NUM_NN_BOTS
    numberOfGreedyBots = parameters.NUM_GREEDY_BOTS
    numberOfBots = numberOfNNBots + numberOfGreedyBots

    if numberOfBots == 0 and not model.viewEnabled:
        modelMustHavePlayers()

    if numberOfHumans != 0:
        createHumans(numberOfHumans, model)

    createBots(numberOfNNBots, model, "NN", parameters, networkLoadPath)
    createBots(numberOfGreedyBots, model, "Greedy", parameters)
    createBots(parameters.NUM_RANDOM_BOTS, model, "Random", parameters)


def process_controller_input(controller, view, model):
    if controller.running:
        controller.process_input()
        return True
    else:
        view.closeView()
        model.set_GUI(False)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" +
              "VIEW WINDOW HAS BEEN CLOSED\n" +
              "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
        return False

def performModelSteps(experience_queue, processNum, model_in_subfolder, loadModel, modelPath, events,
                      weight_manager, guiEnabled, spectate):
    SPEC_OS = importlib.util.find_spec('.networkParameters', package=getPackageName(modelPath))
    parameters = importlib.util.module_from_spec(SPEC_OS)
    SPEC_OS.loader.exec_module(parameters)
    del SPEC_OS
    num_cores = mp.cpu_count()
    if num_cores > 1 and parameters.NUM_COLLECTORS > 1:
        os.sched_setaffinity(0, {(processNum-1)%(num_cores-1)+1})  # Core #1 is reserved for trainer process
    # p = psutil.Process()
    # p.nice(0)

    # Create game instance
    networkLoadPath = modelPath + "models/"
    # TODO: Is this function call needed?
    setSeedAccordingToFolderNumber(model_in_subfolder, loadModel, modelPath, False)
    processInGUISet = processNum in parameters.GUI_COLLECTOR_SET
    if processInGUISet:
        model = Model(guiEnabled, parameters.VIEW_ENABLED, parameters)
    else:
        model = Model(False, False, parameters)
    createModelPlayers(parameters, model, networkLoadPath)
    model.initialize()
    step = 0

    controller = None
    view = None
    # if parameters.ENABLE_TRAINING:
    if processInGUISet:
        if guiEnabled:
            if spectate == 1:
                model.addPlayerSpectator()
            view = View(model, int(SCREEN_WIDTH), int(SCREEN_HEIGHT), parameters)
            controller = Controller(model, parameters.VIEW_ENABLED, view, True)
            view.draw()

    # Move collector to it's required step progression (to break correlations between game stage across collectors)
    print("Desynchronizing worker #" + str(processNum) + "...")
    for i in range((processNum-1) * int(parameters.RESET_LIMIT / parameters.NUM_COLLECTORS)):
        if guiEnabled and processInGUISet:
            guiEnabled = process_controller_input(controller, view, model)
        model.update()
        step += 1
    model.resetBots()
    print("Finished desynchronizing worker #" + str(processNum) + ".")
    events[processNum].set()
    events["Col_can_proceed"].wait()

    # Run game until terminated
    while True:
        for _ in range(parameters.FRAME_SKIP_RATE+2):
            if guiEnabled and processInGUISet:
                guiEnabled = process_controller_input(controller, view, model)
            model.update()
            step += 1
        # After a bot has 'n' amount of experiences, send them to trainer.
        all_experienceLists = [bot.getExperiences() for bot in model.getNNBots()]
        model.resetBots()
        events["Col_can_proceed"].clear()
        if __debug__:
            shape = np.shape(all_experienceLists)
            print("Collector #" + str(processNum) + " is adding to experience queue " +
                  str(shape[0]) + " lists.")
        for experienceList in all_experienceLists:
            if len(experienceList) != 1:
                print("Bot collected the wrong amount of experiences this step (" + str(len(experienceList)) +")")
                print("Quitting...")
                quit()
            experience_queue.put(experienceList)
        events[processNum].set()
        if __debug__:
            print("Collector #" + str(processNum) + " is waiting.")
        events["Col_can_proceed"].wait()
        if __debug__:
            print("Collector #" + str(processNum) + " continued.")
        for bot in model.getNNBots():
            bot.getLearningAlg().setNetworkWeights(weight_manager[0])
            if __debug__ and parameters.VERY_DEBUG:
                for weight in bot.getLearningAlg().getNetworkWeights():
                    m = hashlib.md5(str(bot.getLearningAlg().getNetworkWeights()[weight]).encode('utf-8'))
                    print("Collector" + str(processNum) + " set weights hash: " + m.hexdigest())
        if step > parameters.RESET_LIMIT-parameters.FRAME_SKIP_RATE+2:
            botMasses = []
            for bot in model.getNNBots():
                botMasses.append(bot.getMassOverTime())
                bot.resetMassList()
            print("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n" +
                  "Collector #" + str(processNum) + " had an episode mean mass of " + str(int(np.mean(botMasses))))
            if __debug__:
                  print(str(np.shape(botMasses)) + " in botMasses shape. Sum of masses: "+ str(np.sum(botMasses)))
            print("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
            model.resetModel()
            step = 0

def performGymSteps(experience_queue, processNum, model_in_subfolder, loadModel, modelPath, events,
                      weight_manager, guiEnabled):
    SPEC_OS = importlib.util.find_spec('.networkParameters', package=getPackageName(modelPath))
    parameters = importlib.util.module_from_spec(SPEC_OS)
    SPEC_OS.loader.exec_module(parameters)
    del SPEC_OS
    num_cores = mp.cpu_count()
    if num_cores > 1 and parameters.NUM_COLLECTORS > 1:
        os.sched_setaffinity(0, {(processNum-1)%(num_cores-1)+1})  # Core #1 is reserved for trainer process
    # p = psutil.Process()
    # p.nice(0)

    # Create game instance
    networkLoadPath = modelPath + "models/"
    # TODO: Is this function call needed?
    setSeedAccordingToFolderNumber(model_in_subfolder, loadModel, modelPath, False)
    processInGUISet = processNum in parameters.GUI_COLLECTOR_SET
    env = gym.make(parameters.GAME_NAME)
    learningAlg = None
    if parameters.ALGORITHM == "Q-learning":
        learningAlg = QLearn(parameters)
    elif parameters.ALGORITHM == "CACLA":
        learningAlg = ActorCritic(parameters)
    elif parameters.ALGORITHM == "DPG":
        learningAlg = ActorCritic(parameters)
    elif parameters.ALGORITHM == "SPG":
        learningAlg = ActorCritic(parameters)
    else:
        print("Please enter a valid algorithm.\n")
        quit()
    networks = {}
    networks = learningAlg.initializeNetwork(networkLoadPath, networks)

    # Move collector forward to decorrelate (to break correlations between game stage across collectors)
    print("Desynchronizing worker #" + str(processNum) + "...")
    # if parameters.CNN_REPR:
    #     env.reset()
    #     observation = env.render(mode='rgb_array')
    #     size = (parameters.CNN_INPUT_DIM_1, parameters.CNN_INPUT_DIM_1)
    #     observation = cv2.resize(observation, dsize=size, interpolation=cv2.INTER_CUBIC)
    # else:
    observation = np.array([env.reset()])
        
    step = 0
    rewardSum = 0
    for i in range((processNum-1) * 100):
        if guiEnabled and processInGUISet:
            env.render()
        actionIdx, action = learningAlg.decideMove(observation, False)
        # If environment has discrete actions, but algorithm is continuous, pick action with highest Q(v)
        if parameters.ALGORITHM == "CACLA" or parameters.ALGORITHM == "DPG" or parameters.ALGORITHM == "SPG":
            actionIdx = np.argmax(action)
        observation, reward, done, info = env.step(actionIdx)
        rewardSum += reward
        # if parameters.CNN_REPR:
        #     observation = env.render(mode='rgb_array')
        #     size = (parameters.CNN_INPUT_DIM_1, parameters.CNN_INPUT_DIM_1)
        #     observation = cv2.resize(observation, dsize=size, interpolation=cv2.INTER_CUBIC)
        # else:
        observation = np.array([observation])
        step += 1
        if done:
            if parameters.GAME_NAME[:8] == "CartPole":
                reward = -1
            # if parameters.CNN_REPR:
            #     env.reset()
            #     observation = env.render(mode='rgb_array')
            #     size = (parameters.CNN_INPUT_DIM_1, parameters.CNN_INPUT_DIM_1)
            #     observation = cv2.resize(observation, dsize=size, interpolation=cv2.INTER_CUBIC)
            # else:
            observation = np.array([env.reset()])
            step = 0
            rewardSum = 0
    print("Finished desynchronizing worker #" + str(processNum) + ".")
    events[processNum].set()
    events["Col_can_proceed"].wait()

    # Run game until terminated
    while True:
        actionIdx, action = learningAlg.decideMove(observation)
        if parameters.ALGORITHM in {"CACLA", "DPG", "SPG"}:
            actionIdx = np.argmax(action)
        for _ in range(parameters.FRAME_SKIP_RATE):
            if guiEnabled and processInGUISet:
                env.render()
            new_observation, reward, done, info = env.step(actionIdx)
            new_observation = np.array([new_observation])
            rewardSum += reward
            step += 1
            if done:
                if parameters.GAME_NAME[:8] == "CartPole":
                    reward = -1
                break
        experienceList = [np.array([observation, action, reward, new_observation, None])]
        events["Col_can_proceed"].clear()
        # After collector has 'n' amount of experiences, send them to trainer.
        if __debug__:
            shape = np.shape(experienceList)
            print("Collector #" + str(processNum) + " is adding to experience queue " +
                  str(shape[0]) + " lists.")

        if len(experienceList) != 1:
            print("Collector collected the wrong amount of experiences this step (" + str(len(experienceList)) +")")
            print("Quitting...")
            quit()
        experience_queue.put(experienceList)
        events[processNum].set()
        if __debug__:
            print("Collector #" + str(processNum) + " is waiting.")
        events["Col_can_proceed"].wait()
        if __debug__:
            print("Collector #" + str(processNum) + " continued.")
        learningAlg.setNetworkWeights(weight_manager[0])
        if __debug__ and parameters.VERY_DEBUG:
            for weight in learningAlg.getNetworkWeights():
                m = hashlib.md5(str(learningAlg.getNetworkWeights()[weight]).encode('utf-8'))
                print("Collector" + str(processNum) + " set weights hash: " + m.hexdigest())
        if done:
            print("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n" +
                  "Collector #" + str(processNum) + " had an episode reward of " + str(rewardSum))
            print("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
            observation = env.reset()
            observation = np.array([observation])
            step = 0
            rewardSum = 0
        else:
            observation = new_observation


def startExperienceCollectors(parameters, experience_queue, model_in_subfolder, loadModel, path, events,
                              weight_manager, guiEnabled, spectate):
    startTime = time.time()
    print("*********************************************************************")
    print("Initializing collectors...")
    numWorkers = parameters.NUM_COLLECTORS
    if numWorkers <= 0:
        print("Number of concurrent games must be a positive integer.")
        quit()
    collectors = []
    for processNum in range(1, numWorkers+1):
        if parameters.GAME_NAME == "Agar.io":
            p = mp.Process(target=performModelSteps, args=(experience_queue, processNum, model_in_subfolder, loadModel,
                                                           path, events, weight_manager, guiEnabled, spectate))
        else:
            p = mp.Process(target=performGymSteps, args=(experience_queue, processNum, model_in_subfolder, loadModel,
                                                           path, events, weight_manager, guiEnabled))
        p.start()
        collectors.append(p)
    print("Collectors initialized.")
    print("Initialization time elapsed:   " + str.format('{0:.3f}', time.time() - startTime) + "s")
    print("*********************************************************************")
    return collectors


def terminateExperienceCollectors(collectors):
    if __debug__:
        print("*********************************************************************")
        startTime = time.time()
        print("Terminating collectors...")
    for p in collectors:
        p.terminate()
        # if not p.is_alive():
        p.join(timeout=0.001)
    if __debug__:
        print("Collectors terminated.")
        print("Termination time elapsed:   " + str.format('{0:.3f}', time.time() - startTime) + "s")
        print("*********************************************************************")


def createLearner(parameters, path):
    algorithmName = parameters.ALGORITHM
    learningAlg = None
    if algorithmName == "Q-learning":
        learningAlg = QLearn(parameters)
    elif algorithmName == "CACLA":
        learningAlg = ActorCritic(parameters)
    elif algorithmName == "DPG":
        learningAlg = ActorCritic(parameters)
    elif algorithmName == "SPG":
        learningAlg = ActorCritic(parameters)
    else:
        print("Please enter a valid algorithm.\n")
    learningAlg.initializeNetwork(path)
    return learningAlg


def train_expReplay(parameters, expReplayer, learningAlg, step):
    # TODO: Use GPU
    batch = expReplayer.sample(parameters.MEMORY_BATCH_LEN)

    idxs, priorities, updated_actions = learningAlg.learn(batch, step)
    if parameters.PRIORITIZED_EXP_REPLAY_ENABLED:
        expReplayer.update_priorities(idxs, np.abs(priorities) + 1e-4)
        if parameters.OCACLA_REPLACE_TRANSITIONS:
            if updated_actions is not None:
                expReplayer.update_dones(idxs, updated_actions)
            else:
                print("Updated actions is None!")


# Train for 'MAX_TRAINING_STEPS'. Meanwhile send signals back to master process to notify of training process.
def trainOnExperiences(experience_queue, collector_events, path, queue, weight_manager):
    SPEC_OS = importlib.util.find_spec('.networkParameters', package=getPackageName(path))
    parameters = importlib.util.module_from_spec(SPEC_OS)
    SPEC_OS.loader.exec_module(parameters)
    del SPEC_OS
    # Increase priority of this process
    num_cores = mp.cpu_count()
    if num_cores > 1 and parameters.NUM_COLLECTORS > 2:
        os.sched_setaffinity(0, {0})  # Core #0 is reserved for trainer process
    # p = psutil.Process()
    # p.nice(0)
    networkPath = path + "models/"
    learningAlg = createLearner(parameters, networkPath)
    weight_manager.append(learningAlg.getNetworkWeights())
    if __debug__ and parameters.VERY_DEBUG:
        for weight in learningAlg.getNetworkWeights():
            m = hashlib.md5(str(learningAlg.getNetworkWeights()[weight]).encode('utf-8'))
            print("Trainer saved weights hash: " + m.hexdigest())
    collector_events["Col_can_proceed"].set()
    if parameters.EXP_REPLAY_ENABLED:
        if parameters.PRIORITIZED_EXP_REPLAY_ENABLED:
            expReplayer = PrioritizedReplayBuffer(parameters.MEMORY_CAPACITY, parameters.MEMORY_ALPHA, parameters.MEMORY_BETA)
        else:
            expReplayer = ReplayBuffer(parameters.MEMORY_CAPACITY)
        # TODO: Uncomment for Anton's LSTM expReplay stuff
        # expReplayer = ExpReplay(parameters)

        # Collect enough experiences before training
        print("\n******************************************************************")
        print("Beginning initial experience collection...")
        collectionTime = time.time()
        while len(expReplayer) < parameters.NUM_EXPS_BEFORE_TRAIN:
            print(str(len(expReplayer)) + " | " + str(parameters.NUM_EXPS_BEFORE_TRAIN), end="\r")
            for experience in experience_queue.get():
                for i in range(1, parameters.NUM_COLLECTORS + 1):
                    collector_events[i].wait()
                    collector_events[i].clear()
                collector_events["Col_can_proceed"].set()
                expReplayer.add(*experience)
            if __debug__:
                print("Buffer size: " + str(len(expReplayer)) + " | " + str(parameters.NUM_EXPS_BEFORE_TRAIN))
        # TODO: Start with buffer completely full?
        # TODO: can experiences be added in batch in Prioritized Replay Buffer?
        print("Initial experience collection completed.")
        print("Current replay buffer size:   ", len(expReplayer))
        print("Collection time elapsed:      " + str.format('{0:.3f}', time.time() - collectionTime) + "s")
        print("\n******************************************************************")
    print("\nBeggining to train...")
    print("////////////////////////////////////////////////////////////////////\n")

    smallPart = max(int(parameters.MAX_TRAINING_STEPS / 100), 1)  # Get int value closest to to 1% of training time
    testInterval = smallPart * parameters.TRAIN_PERCENT_TEST_INTERVAL
    targNet_stepChunk = parameters.TARGET_NETWORK_STEPS
    printSteps = 500
    timeStep = time.time()
    if parameters.GAME_NAME[:8] == "CartPole":
        cartPoleTest(learningAlg, path, "start.png", parameters)

    for step in range(parameters.CURRENT_STEP, parameters.MAX_TRAINING_STEPS):
        if step % printSteps == 0:
            test_stepsLeft = testInterval - (step % testInterval)
            targNet_stepsLeft = targNet_stepChunk - (step % targNet_stepChunk)
            elapsedTime = time.time() - timeStep
            print("____________________________________________________________________\n" +
                  "Steps before next test copy:                " + str(test_stepsLeft) + " | Total: " + str(testInterval) + "\n" +
                  "Steps before target network update:         " + str(targNet_stepsLeft) + " | Total: " + str(targNet_stepChunk) + "\n" +
                  "Steps performed:                            " + str(step) + " | Total: " + str(parameters.MAX_TRAINING_STEPS) + "\n" +
                  "Current exploration:                        " + str(learningAlg.getNoise()))
            if parameters.EXP_REPLAY_ENABLED:
                print("Current replay buffer size:                 " + str(len(expReplayer)) + " | Total: " + str(parameters.MEMORY_CAPACITY))
            print("Time elapsed during last " + str(printSteps) + " train steps:  " + str.format('{0:.3f}', elapsedTime) +
                  "s   (" + str.format('{0:.3f}', elapsedTime*1000/printSteps) + "ms/step)\n" +
                  "¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")
            timeStep = time.time()
        # Train 1 step
        batch = []
        while len(batch) < parameters.NUM_COLLECTORS*parameters.NUM_NN_BOTS:
            batch.append(experience_queue.get())

        # Signal collectors to collect another set of experiences
        if parameters.ONE_BEHIND_TRAINING:
            for i in range(1, parameters.NUM_COLLECTORS + 1):
                collector_events[i].wait()
                collector_events[i].clear()
            collector_events["Col_can_proceed"].set()
        if __debug__:
            print("Trainer received a lists of " + str(len(batch)) + " exps.")

        #TODO: Does AC have target netowrks?
        if parameters.EXP_REPLAY_ENABLED:
            for experience in batch:
                expReplayer.add(*(experience[0]))
            train_expReplay(parameters, expReplayer, learningAlg, step)
        else:
            tr_batch = np.array(batch).transpose()
            batch = (tr_batch[0,0,:], tr_batch[1,0,:], tr_batch[2,0,:], tr_batch[3,0,:], tr_batch[4,0,:])
            _,_,_ = learningAlg.learn(batch, step)

        weight_manager[0] = learningAlg.getNetworkWeights()
        if __debug__ and parameters.VERY_DEBUG:
            for weight in learningAlg.getNetworkWeights():
                m = hashlib.md5(str(learningAlg.getNetworkWeights()[weight]).encode('utf-8'))
                print("Trainer saved weights hash: " + m.hexdigest())

        # Signal collectors to collect another set of experiences
        if not parameters.ONE_BEHIND_TRAINING:
            for i in range(1, parameters.NUM_COLLECTORS + 1):
                collector_events[i].wait()
                collector_events[i].clear()
            collector_events["Col_can_proceed"].set()

        if (step+1) % testInterval == 0:
            if __debug__:
                print("Copying network to new file '" + str(parameters.CURRENT_STEP + testInterval) + "_model.h5 ...")
            learningAlg.save(path)
            learningAlg.save(path, str(step + 1) + "_")
            params = learningAlg.getUpdatedParams()
            tweakedTotal = [[paramName, params[paramName], checkValidParameter(paramName)] for paramName in params]
            tweakedTotal.append(["CURRENT_STEP", step+1, checkValidParameter("CURRENT_STEP")])
            modifyParameterValue(tweakedTotal, path)
        # Check if we should print the training progress percentage
        if (step+1) % smallPart == 0:
            if __debug__:
                print("Trainer sending signal: 'PRINT_TRAIN_PROGRESS'")
            queue.put("PRINT_TRAIN_PROGRESS")
    # Signal that training has finished
    if __debug__:
        print("Trainer sending signal: 'DONE'")
    queue.put("DONE")
    if parameters.GAME_NAME[:8] == "CartPole":
        cartPoleTest(learningAlg, path, "end.png", parameters)


def cartPoleTest(alg, path, name, parameters):
    x = []
    y = []
    z = []
    for v in range(3*5):
        y.append(str(float(v/5.0-1.5)))
        row = []
        for t in range(30*5):
            if parameters.ALGORITHM == "Q-learning":
                network = alg.getNetwork()
                values = network.predict_action(np.array([[0.0,0.0,float(3.14/180*t/5 -3.14/180*15),float(v/5.0-1.5)]]))
                row.append(values[np.argmax(values)])

            elif parameters.ALGORITHM == "CACLA":
                network = alg.getNetworks()["V(S)"]
                value = network.predict(np.array([[0.0,0.0,float(3.14/180*t/5 -3.14/180*15),float(v/5.0-1.5)]]))
                row.append(value)
                
            elif parameters.ALGORITHM in {"DPG", "SPG"}:
                network = alg.getNetworks()["Q(S,A)"]
                state = np.array([[0.0,0.0,float(3.14/180*t/5 -3.14/180*15),float(v/5.0-1.5)]])
                action = np.array([[0.0, 0.0]])
                value = network.predict(state,action)
                row.append(value)
                
            else:
                print("You w0t m8?")
                quit()
        z.append(np.array(row))

    for t in range(3*5):
        x.append(str(float(3.14/180*t/5 -3.14/180*15)))

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    plt.imshow(z, extent=[-0.26,0.26,-3.1,3.1], aspect="auto")
    plt.xlabel("Angle (rad)")
    plt.ylabel("Velocity (rad/time)")
    plt.savefig(name)

# Create the asynchronous training procedure.
# The experience replay buffer is created as a multiprocessing.mp.Manager. This manager takes form as the expReplay class
# and is shared across all subprocesses. In order for subprocesses to access methods of the class, a Proxy manager needs
# to be made.
# The replay_buffer is first filled with enough experiences to begin training. Then, the training happens asynchronously
# from the experience collection. 'N' amount of training processes are initialized and train while 'X' amount of
# subprocesses collect experiences. After a given amount of training steps, collector subprocesses are killed and
# re-initialized with a more up-to-date version of the network.
# Every a certain amount of training, testing is done. This also happens with the training status being printed.
def trainingProcedure(parameters, model_in_subfolder, loadModel, path, startTime, guiEnabled, spectate):
    # Perform simultaneous experience collection and training
    currentPart = parameters.CURRENT_STEP
    smallPart = max(int(parameters.MAX_TRAINING_STEPS / 100), 1)  # Get int value closest to to 1% of training time
    trainInterval = smallPart * parameters.TRAIN_PERCENT_TEST_INTERVAL
    num_cores = mp.cpu_count()
    print(num_cores)
    if num_cores > 1 and parameters.NUM_COLLECTORS > 1:
        os.sched_setaffinity(0, {num_cores-1})  # Core #0 is reserved for trainer process
    while currentPart < parameters.MAX_TRAINING_STEPS:
        # Create collectors and exp queue
        experience_queue = mp.Queue()
        collector_events = {"Col_can_proceed": mp.Event()}
        for i in range(1,parameters.NUM_COLLECTORS+1):
            collector_events[i] = mp.Event()
        weight_manager = mp.Manager().list()
        collectors = startExperienceCollectors(parameters, experience_queue, model_in_subfolder, loadModel, path,
                                               collector_events, weight_manager, guiEnabled, spectate)
        for i in range(1, NUM_COLLECTORS + 1):
            collector_events[i].wait()
            collector_events[i].clear()
        # Create training process and communication pipe
        trainer_master_queue = mp.Queue()
        # TODO: Create multiple learners
        trainer = mp.Process(target=trainOnExperiences, args=(experience_queue, collector_events, path, trainer_master_queue, weight_manager))
        trainer.start()
        # Training is split into periods. In case there is a training failure or early stop, training progress will be
        # re-started from previous save point.
        # Begin training period
        print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" + "\n" +
              "Training period " + str(currentPart) + "-->" + str(currentPart + trainInterval) + "\n" +
              "Began: " + elapsedTimeText(time.time() - startTime) + "\n" +
              "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
        while True:
            # While trainer has sent no signal, check if trainer should be terminated
            while trainer_master_queue.empty():
                sleep(1)
            # Get a signal from trainer
            trainer_signal = trainer_master_queue.get()
            # Print when 1% of training time has elapsed
            if trainer_signal == "PRINT_TRAIN_PROGRESS":
                if __debug__:
                    print("Master received signal: 'PRINT_TRAIN_PROGRESS'")
                currentPart = printTrainProgress(parameters, currentPart, startTime)
            # When trainer signals training is 'DONE', terminate child processes (collectors and trainer) and break
            if trainer_signal == "DONE":
                if __debug__:
                    print("Master received signal: 'DONE'")
                trainer.join(timeout=0.001)
                terminateExperienceCollectors(collectors)
                break
        print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n" +
              "Training period ended.\n" +
              "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")

    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Training done.\n")


def run():
    guiEnabled = int(input("Enable GUI?: (1 == yes)\n"))
    guiEnabled = (guiEnabled == 1)

    tweakedTotal = []
    modelName = None
    modelPath = None
    loadedModelName = None
    packageName = None
    model_in_subfolder = False
    loadModel = int(input("Do you want to load a model? (1 == yes)\n"))
    loadModel = (loadModel == 1)
    if loadModel:
        while packageName is None:
            packageName = None
            print("#########################################")
            print("Saved Models: \n")
            for folder in [i for i in os.listdir("savedModels/")]:
                print(folder)
            modelName = input("Enter the model name (name of directory in savedModels): (Empty string == break)\n")
            # If user presses enter, quit model loading
            if str(modelName) == "":
                loadModel = False
                modelName = None
                break
            # If user inputs wrong model name, ask for input again
            modelPath = "savedModels/" + modelName + "/"
            if not os.path.exists(modelPath):
                print("Invalid model name, no model found under ", modelPath)
                continue
            # CHECK FOR SUBFOLDERS
            if str(modelName)[0] != "$":
                while packageName is None:
                    print("------------------------------------")
                    print("Folder Submodels: \n")
                    for folder in [i for i in os.listdir(modelPath) if os.path.isdir(modelPath + "/" + i)]:
                        print(folder)
                    subModelName = input("Enter the submodel name: (Empty string == break)\n")
                    # If user presses enter, leave model
                    if str(subModelName) == "":
                        break
                    subPath = modelPath + subModelName + "/"
                    if not os.path.exists(subPath):
                        print("Invalid model name, no model found under ", subPath)
                        continue
                    packageName = "savedModels." + modelName + "." + subModelName
                    loadedModelName = subPath
                    # modelName = path
                    model_in_subfolder = True
                if packageName is None:
                    continue

            if packageName is None:
                packageName = "savedModels." + modelName
                loadedModelName = modelPath
                # ModelName = None will autogenereate a name
                modelName = None

    # TODO: Check if algorithm loads correctly (both when starting fresh train and when loading pretrained model)
    tweaking = int(input("Do you want to tweak parameters? (1 == yes)\n"))
    if tweaking == 1:
        while True:
            tweakedParameter = str(input("Enter name of parameter to be tweaked:\n"))
            paramLineNumber = checkValidParameter(tweakedParameter)
            if paramLineNumber is not None:
                paramValue = str(input("Enter parameter value:\n"))
                tweakedTotal.append([tweakedParameter, paramValue, paramLineNumber])
            if 1 != int(input("Tweak another parameter? (1 == yes)\n")):
                break
        modelPath = "savedModels/" + nameSavedModelFolder(tweakedTotal)
        model_in_subfolder = True

    if int(input("Give saveModel folder a custom name? (1 == yes)\n")) == 1:
        modelPath = "savedModels/" + str(input("Input folder name:\n"))

    modelPath, startTime = initModelFolder(modelPath, loadedModelName, model_in_subfolder)
    print("Initialized folder: " + modelPath)

    if tweakedTotal:
        modifyParameterValue(tweakedTotal, modelPath)
    packageName = getPackageName(modelPath)
    print("Import package name: " + packageName)
    parameters = importlib.import_module(packageName +'.networkParameters')
    # Initialize network while number of humans is determined
    if parameters.CURRENT_STEP == 0:
        print("\nInitializing network...\n")
        p = mp.Process(target=createNetwork, args=(parameters, modelPath))
        p.start()
        p.join()

    # Determine number of humans
    numberOfHumans = 0
    spectate = 0
    if parameters.GAME_NAME == "Agar.io":
        if guiEnabled:
            numberOfHumans = int(input("Please enter the number of human players: (" + str(MAXHUMANPLAYERS) + " max)\n"))
        spectate = None
        if guiEnabled and not numberOfHumans > 0:
            spectate = int(input("Do want to spectate an individual bot's FoV? (1 = yes)\n")) == 1

    if numberOfHumans == 0:
        startTime = time.time()
        if parameters.ENABLE_TRAINING:
            trainingProcedure(parameters, model_in_subfolder, loadModel, modelPath, startTime, guiEnabled, spectate)
            print("--------")
            print("Training time elapsed:               " + elapsedTimeText(int(time.time()- startTime)))
            print("Average time per update:    " +
                  str.format('{0:.3f}', (time.time() - startTime) / parameters.MAX_TRAINING_STEPS) + " seconds")
            print("--------")
        if parameters.ENABLE_TESTING:
            testStartTime = time.time()
            # Post train testing
            if parameters.FINAL_TEST_NUM > 0:
                print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("Performing post-training tests...\n")
                if parameters.GAME_NAME == "Agar.io":
                    runFinalTests(modelPath, parameters)
                else:
                    runFinalGymTests(modelPath,parameters)
                print("\nFinal tests completed.\n")
            if parameters.DUR_TRAIN_TEST_NUM > 0:
                # During train testing
                print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("Performing during training tests...")
                testResults = []
                smallPart = max(int(parameters.MAX_TRAINING_STEPS / 100), 1)  # Get int value closest to to 1% of training time
                for testPercent in range(0, 101, parameters.TRAIN_PERCENT_TEST_INTERVAL):
                    testName = str(testPercent) + "%"
                    testStepNumber = testPercent * smallPart
                    testNetworkPath = modelPath + "models/" + str(testStepNumber) + "_"
                    if parameters.GAME_NAME == "Agar.io":
                        testResults.append(testingProcedure(modelPath, testNetworkPath, parameters, testName,
                                                            parameters.DUR_TRAIN_TEST_NUM)[0])
                    else:
                        testResults.append(gymTestingProcedure(modelPath, testNetworkPath, parameters, testName,
                                                            parameters.DUR_TRAIN_TEST_NUM)[0])
                exportTestResults(testResults, modelPath, parameters)
                print("\nDuring training tests completed.")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            modelPath = finalPathName(parameters, modelPath)
            print("--------")
            print("Testing time elapsed:               " + elapsedTimeText(int(time.time()- testStartTime)))
            print("--------")
        if model_in_subfolder:
            print(os.path.join(modelPath))
            createCombinedModelGraphs(os.path.join(modelPath))
        print("--------")
        print("Total time elapsed:               " + elapsedTimeText(int(time.time() - startTime)))
        print("--------")


if __name__ == '__main__':
    run()
