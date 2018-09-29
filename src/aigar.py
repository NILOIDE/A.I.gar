import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #This suppresses tensorflow AVX warnings
import sys
import importlib
import importlib.util
import shutil
import time
import datetime
#import pyximport; pyximport.install()
from controller.controller import Controller
from model.qLearning import *
from model.actorCritic import *
from model.bot import *
from model.model import Model
from model.expReplay import ExpReplay
from model.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import matplotlib.pyplot as plt
from builtins import input
import pathos.multiprocessing as mp
from multiprocessing import Process, Pipe
from multiprocessing.managers import BaseManager, NamespaceProxy
from time import sleep

from view.view import View
from modelCombiner import createCombinedModelGraphs, plot


import numpy as np
# import tensorflow as tf
import random as rn


def fix_seeds(seedNum):
    import tensorflow as tf

    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

    # import os
    # os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
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
    copyParameters(path, loadedModelName)
    return path, startTime


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
    time.sleep(numpy.random.rand())
    if not os.path.exists(osPath):
        os.makedirs(osPath)
    # Create folder based on name
    now = datetime.datetime.now()
    startTime = now
    nowStr = now.strftime("%b-%d_%H:%M:%S:%f")
    path = superName + "$" + nowStr + "$"
    time.sleep(numpy.random.rand())
    if os.path.exists(path):
        randNum = numpy.random.randint(100000)
        path = superName + "$" + nowStr + "-" + str(randNum) + "$"
    os.makedirs(path)
    path += "/"
    print("Super Path: ", superName)
    print("Path: ", path)
    return path, startTime

def createNamedPath(self, superName):
    #Create savedModels folder
    basePath = "savedModels/"
    if not os.path.exists(basePath):
        os.makedirs(basePath)
    #Create subFolder for given parameter tweaking
    osPath = os.getcwd() + "/" + superName
    time.sleep(numpy.random.rand())
    if not os.path.exists(osPath):
        os.makedirs(osPath)
    #Create folder based on name
    now = datetime.datetime.now()
    startTime = now
    nowStr = now.strftime("%b-%d_%H:%M:%S:%f")
    path = superName  + "$" + nowStr + "$"
    time.sleep(numpy.random.rand())
    if os.path.exists(path):
        randNum = numpy.random.randint(100000)
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
            print("FOUND")
            return n
    #print("Parameter with name " + tweakedParameter + "not found.")
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
        print(tweaked[i][0])
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


def createNetwork(parameters, path, name=""):
    network = Network(parameters)
    network.saveModel(path, name)


def createHumans(numberOfHumans, model1):
    for i in range(numberOfHumans):
        name = input("Player" + str(i + 1) + " name:\n")
        model1.createHuman(name)


def createBots(number, model, botType, parameters, loadPath=None):
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
            else:
                print("Please enter a valid algorithm.\n")
                quit()

            networks = learningAlg.initializeNetwork(loadPath, networks)
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
    # TODO: Change test's RESET_LIMIT to 0? (Would get rid of resetting at the end of episode)
    testParameters.GATHER_EXP = False
    if virus != None:
        testParameters.VIRUS_SPAWN = virus
    testParameters.NUM_NN_BOTS = num_nn_bots
    testParameters.NUM_GREEDY_BOTS = num_greedy_bots
    testParameters.NUM_RAND_BOTS = num_rand_bots
    return testParameters


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

        meanY = numpy.mean(y)
        ax.legend(loc='upper left')
        ax.set_ylabel(yLabel)
        ax.set_title(title + " mean value (" + str(round(meanY, 1)) + ") $\pm$ $\sigma$ interval")
        ax.grid()
        fig.savefig(path + title + ".pdf")

        plt.close()


# Plot test results and export data
def exportTestResults(testResults, path, parameters, testInterval):
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

    plotTesting(testResults, path, testInterval, maxSteps)


# Perform 1 episode of the test. Return the mass over time list, the mean mass of the episode, and the max mass.
def performTest(path, testParams, testSteps):
    testModel = Model(False, False, testParams)
    createModelPlayers(testParams, testModel, path)
    testModel.initialize(False)
    for step in range(testSteps):
        testModel.update()

    bots = testModel.getBots()
    massOverTime = [bot.getMassOverTime() for bot in bots]
    meanMass = numpy.mean([numpy.mean(botMass) for botMass in massOverTime])
    maxMeanMass = numpy.max(meanMass)
    maxMass = numpy.max([numpy.max(botMass) for botMass in massOverTime])
    varianceMass = numpy.mean(numpy.var(massOverTime))

    print("Test run:" +
          "\n  Process id: " + str(os.getpid()) +
          "\n  Number of bot mass lists: " + str(len(massOverTime)) +
          "\n  Mean mass: "+ str(meanMass) +
          "\n  Max mass: "+ str(maxMass) +
          "\n")

    return [massOverTime, meanMass, maxMass]


# Test the model for 'n' amount of episodes for the given type of test. This is done in parallel uses a pool of workers.
# The test results from all tests are put together into 1 structure to then be used for averaging and plotting.
def testModel(path, name, plotName, testParams, n_tests):
    print("\nTesting", name, "...")
    print("------------------------------------------------------------------\n")

    start = time.time()
    testSteps = testParams.RESET_LIMIT
    # TODO: Parallel testing used to take as long as serial. Test this.
    # Serial Testing
    # testResults = []
    # for i in range(n_tests):
    #     testResults.append(performTest(path, testParams, testSteps))

    # Parallel testing

    #TODO test wether we actually NEED pathos.multiprocessing or if we can do just with OG multiprocessing
    pool = mp.Pool(n_tests)
    print("Initializing " + str(n_tests) + " testers..." )
    testResults = pool.starmap(performTest, [(path, testParams, testSteps) for process in range(n_tests)])
    pool.close()
    pool.join()
    print("Number of tests: ", n_tests, "Time elapsed: ", time.time() - start, "\n")
    print("------------------------------------------------------------------\n")

    masses = []
    meanMasses = []
    maxMasses = []
    for test in range(n_tests):
        masses.extend(testResults[test][0])
        meanMasses.append(testResults[test][1])
        maxMasses.append(testResults[test][2])
    evals = {"name":name}
    evals["plotName"] = plotName
    evals["meanScore"] = numpy.mean(meanMasses)
    evals["stdMean"] = numpy.std(meanMasses)
    evals["meanMaxScore"] = numpy.mean(maxMasses)
    evals["stdMax"] = numpy.std(maxMasses)
    evals["maxScore"] = numpy.max(maxMasses)
    # return (name, maxScore, meanScore, stdMean, meanMaxScore, stdMax), masses
    return evals, masses


def testingProcedure(path, parameters, packageName, n_tests):
    # currentAlg = model.getNNBot().getLearningAlg()
    # originalNoise = currentAlg.getNoise()
    # currentAlg.setNoise(0)
    #
    # originalTemp = None
    # if str(currentAlg) != "AC":
    #     originalTemp = currentAlg.getTemperature()
    #     currentAlg.setTemperature(0)
    # TODO: Perform all test kinds simultaneously
    testEvals = {}
    masses = {}
    testParams = createTestParams(packageName)
    testEvals["current"], masses["current"] = testModel(path, "test", "Test", testParams, n_tests)

    pelletTestParams = createTestParams(packageName, False)
    testEvals["pellet"], masses["pellet"] = testModel(path, "pellet", "Pellet_Collection", pelletTestParams, n_tests)

    if parameters.MULTIPLE_BOTS_PRESENT:
        greedyTestParams = createTestParams(packageName, False, 1, 1)
        testEvals["vsGreedy"], masses["vsGreedy"] = testModel(path, "vsGreedy", "Vs_Greedy", greedyTestParams, n_tests)

    if parameters.VIRUS_SPAWN:
        virusTestParams = createTestParams(packageName, True)
        testEvals["virus"], masses["virus"] = testModel(path, "pellet_with_virus", "Pellet_Collection_with_Viruses",
                                                        virusTestParams, n_tests)
        if parameters.MULTIPLE_BOTS_PRESENT:
            virusGreedyTestParams = createTestParams(packageName, True, 1, 1)
            testEvals["virusGreedy"], masses["virusGreedy"] = testModel(path, "vsGreedy_with_virus", "Vs_Greedy_with_Viruses",
                                                                        virusGreedyTestParams, n_tests)

    # TODO: Check if following commented noise code is needed
    # currentAlg.setNoise(originalNoise)
    #
    # if str(currentAlg) != "AC":
    #     currentAlg.setTemperature(originalTemp)

    return testEvals, masses


def runFinalTests(path, parameters, packageName):
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Performing final tests...\n")
    evals, masses = testingProcedure(path, parameters, packageName, parameters.FINAL_TEST_LEN)

    # TODO: add more test scenarios for multiple greedy bots and full model check

    name_of_file = path + "/final_results.txt"
    with open(name_of_file, "w") as file:
        data = "Number of runs per testing: " + str(parameters.FINAL_TEST_LEN) + "\n"
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
            meanVal = val / parameters.FINAL_TEST_LEN
            meanMassPerTimeStep.append(meanVal)
        # exportTestResults(meanMassPerTimeStep, modelPath, "Mean_Mass_" + name)
        labels = {"meanLabel": "Mean Mass", "sigmaLabel": '$\sigma$ range', "xLabel": "Step number",
                  "yLabel": "Mass mean value", "title": "Mass plot test phase", "path": path,
                  "subPath": "Mean_Mass_" + str(evals[testType]["plotName"])}
        plot(masses[testType], parameters.RESET_LIMIT, 1, labels)
    print("\nTesting completed.")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")


def performGuiModel(parameters, enableTrainMode, loadedModelName, model_in_subfolder, loadModel, modelPath, algorithm,
                    guiEnabled, viewEnabled, numberOfHumans, spectate):

    mouseEnabled = True
    humanTraining = False
    if fitsLimitations(numberOfHumans, MAXHUMANPLAYERS):
        # createHumans(numberOfHumans, model)
        if numberOfHumans > 0 and not humanTraining:
            mouseEnabled = int(input("Do you want control Player1 using the mouse? (1 == yes)\n"))

    model = Model(guiEnabled, viewEnabled, parameters, True)
    model.setTrainingEnabled(enableTrainMode == 1)
    if spectate == 1:
        model.addPlayerSpectator()

    createModelPlayers(parameters, model, numberOfHumans, algorithm, loadModel)

    Bot.init_exp_replayer(parameters, loadedModelName)
    setSeedAccordingToFolderNumber(model_in_subfolder, loadModel, modelPath, enableTrainMode)

    model.initialize(loadModel)
    screenWidth, screenHeight = defineScreenSize(numberOfHumans)

    testResults = None
    if guiEnabled:
        view = View(model, screenWidth, screenHeight, parameters)
        controller = Controller(model, viewEnabled, view, mouseEnabled)
        view.draw()
        while controller.running:
            controller.process_input()
            model.update()
    else:
        maxSteps = parameters.MAX_SIMULATION_STEPS
        smallPart = max(int(maxSteps / 100), 1)  # constitutes one percent of total training time
        testPercentage = smallPart * 5

        for step in range(maxSteps):
            model.update()
            if step % smallPart == 0 and step != 0:
                print("Trained: ", round(step / maxSteps * 100, 1), "%")
                # Test every 5% of training
            if parameters.ENABLE_TESTING:
                if step % testPercentage == 0:
                    testResults = updateTestResults(testResults, model, round(step / maxSteps * 100, 1), parameters)


def createModelPlayers(parameters, model, path=None, numberOfHumans=0):
    numberOfNNBots = parameters.NUM_NN_BOTS
    numberOfGreedyBots = parameters.NUM_GREEDY_BOTS
    numberOfBots = numberOfNNBots + numberOfGreedyBots

    if numberOfBots == 0 and not model.viewEnabled:
        modelMustHavePlayers()

    if numberOfHumans != 0:
        createHumans(numberOfHumans, model)

    createBots(numberOfNNBots, model, "NN", parameters, path)
    createBots(numberOfGreedyBots, model, "Greedy", parameters)
    createBots(parameters.NUM_RANDOM_BOTS, model, "Random", parameters)


def addExperiencesToBuffer(expReplayer, gatheredExperiences, processNum):
    if __debug__:
        print("Experiences shape of collector #" + str(processNum) + ": ", np.shape(gatheredExperiences), "\n")
    for experienceList in gatheredExperiences:
        for experience in experienceList:
            # TODO: Make add method nicer by taking entire memory as argument?
            expReplayer.add(experience[0], experience[1], experience[2], experience[3], experience[4])
    return expReplayer


def performModelSteps(parameters, expReplayer, processNum, model_in_subfolder, loadModel, modelPath):
    # Create game instance
    model = Model(False, False, parameters)
    createModelPlayers(parameters, model, modelPath)
    # TODO: Is this function call needed?
    setSeedAccordingToFolderNumber(model_in_subfolder, loadModel, modelPath, False)
    model.initialize(loadModel)
    # Run game until terminated
    while True:
        for step in range(parameters.RESET_LIMIT):
            model.update()
        gatheredExperiences = [bot.getExperiences() for bot in model.getBots()]
        addExperiencesToBuffer(expReplayer, gatheredExperiences, processNum)
        # TODO: We can also completely reinitialize the game (maybe OpenAI gym doesn't allow game resetting)
        model.resetModel()


def startExperienceCollectors(parameters, expReplayer, loadedModelName, model_in_subfolder, loadModel, path):
    numWorkers = parameters.NUM_COLLECTORS
    if numWorkers <= 0:
        print("Number of concurrent games must be a positive integer.")
        quit()
    collectors = []
    for processNum in range(numWorkers):
        p = Process(target=performModelSteps, args=(parameters, expReplayer, processNum, model_in_subfolder, loadModel, path))
        p.start()
        collectors.append(p)
    return collectors


def terminateExperienceCollectors(collectors):
    print("Terminating collectors...")
    for p in collectors:
        p.terminate()
        # if not p.is_alive():
        p.join(timeout=0.001)


class MyManager(BaseManager): pass

class TestProxy(NamespaceProxy):
    # TODO: Make compatible with Prioritized exp replay
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', '__len__', 'add', 'sample', 'updatePriorities' )

    def add(self, obs_t, action, reward, obs_tp1, done):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.add.__name__, (obs_t, action, reward, obs_tp1, done))

    def __len__(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.__len__.__name__)

    def sample(self, batch_size):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.sample.__name__, (batch_size,))


def createLearner(parameters, path):
    algorithmName = parameters.ALGORITHM
    learningAlg = None
    if algorithmName == "Q-learning":
        learningAlg = QLearn(parameters)
    elif algorithmName == "CACLA":
        learningAlg = ActorCritic(parameters)
    learningAlg.initializeNetwork(path)
    return learningAlg


def train(parameters, expReplayer, learningAlg):
    # TODO: Use GPU
    batch = expReplayer.sample(parameters.MEMORY_BATCH_LEN)
    idxs, priorities, updated_actions = learningAlg.learn(batch)
    if parameters.PRIORITIZED_EXP_REPLAY_ENABLED:
        expReplayer.update_priorities(idxs, numpy.abs(priorities) + 1e-4)
        if parameters.OCACLA_REPLACE_TRANSITIONS:
            if updated_actions is not None:
                expReplayer.update_dones(idxs, updated_actions)
            else:
                print("Updated actions is None!")


# Train for 'MAX_TRAINING_STEPS'. Meanwhile send signals back to master process to notify of training process.
def trainOnExperiences(parameters, expReplayer, path, connection):
    learningAlg = createLearner(parameters, path)
    smallPart = max(int(parameters.MAX_TRAINING_STEPS / 100), 1)  # Get int value closest to to 1% of training time
    testInterval = smallPart * parameters.TRAIN_PERCENT_TEST_INTERVAL
    stepChunk = parameters.NETWORK_UPDATE_STEPS
    for step in range(parameters.MAX_TRAINING_STEPS):
        if __debug__:
            print("Current replay buffer size: ", len(expReplayer))
            print("Steps before target network update: ", step, "Target network update interval: ", stepChunk)
        # Check if testing should happen
        if parameters.ENABLE_TESTING and step % testInterval == 0:
            connection.send("TEST")
        # Check if collectors should have their networks reloaded
        if step != 0 and step % stepChunk == 0:
            connection.send("RELOAD_COLLECTOR")

        train(parameters, expReplayer, learningAlg)
        # Check if we should print the training progress percentage
        if step != 0 and step % smallPart == 0:
            connection.send("PRINT_TRAIN_PROGRESS")
    # Signal that training has finished
    connection.send("DONE")


# Create the asynchronous training procedure.
# The experience replay buffer is created as a multiprocessing.Manager. This manager takes form as the expReplay class
# and is shared across all subprocesses. In order for subprocesses to access methods of the class, a Proxy manager needs
# to be made.
# The replay_buffer is first filled with enough experiences to begin training. Then, the training happens asynchronously
# from the experience collection. 'N' amount of training processes are initialized and train while 'X' amount of
# subprocesses collect experiences. After a given amount of training steps, collector subprocesses are killed and
# re-initialized with a more up-to-date version of the network.
# Every a certain amount of training, testing is done. This also happens with the training status being printed.
def trainingProcedure(parameters, loadedModelName, model_in_subfolder, loadModel, path, packageName):
    # TODO: check if I implemented ExpReplayer correctly
    if parameters.PRIORITIZED_EXP_REPLAY_ENABLED:
        MyManager.register('ExpReplayer', PrioritizedReplayBuffer, TestProxy)
        manager = MyManager()
        manager.start()
        expReplayer = manager.ExpReplayer(parameters.MEMORY_CAPACITY, parameters.MEMORY_ALPHA, parameters.MEMORY_BETA)
    else:
        MyManager.register('ExpReplayer', ReplayBuffer, TestProxy)
        manager = MyManager()
        manager.start()
        expReplayer = manager.ExpReplayer(parameters.MEMORY_CAPACITY)

    # TODO: Uncomment for Anton's LSTM expReplay stuff
    # expReplayer = ExpReplay(parameters)


    # Perform simultaneous experiencing and training
    testResults = []
    collectors = None
    smallPart = max(int(parameters.MAX_TRAINING_STEPS / 100), 1)  # Get int value closest to to 1% of training time
    currentPart = 0
    testInterval = smallPart * parameters.TRAIN_PERCENT_TEST_INTERVAL
    # TODO: Create multiple learners
    # Create training process and communication pipe
    master_conn, trainer_conn = Pipe()
    trainer = Process(target=trainOnExperiences, args=(parameters, expReplayer, path, trainer_conn))
    # Gather initial experiences
    print("\n******************************************************************")
    print("Beggining to initial experience collection...\n")
    collectors = startExperienceCollectors(parameters, expReplayer, loadedModelName, model_in_subfolder, loadModel,
                                           path)
    # TODO: Start with buffer completely full?
    # TODO: can experiences be added in batch in Prioritized Replay Buffer?
    # Collect enough experiences before training
    while len(expReplayer) < parameters.NUM_EXPS_BEFORE_TRAIN:
        pass
    print("Initial experience collection completed.")
    print("Current replay buffer size: ", len(expReplayer))
    print("\nBeggining to train...")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    trainer.start()
    while True:
        # Check if trainer has sent a signal threw the pipe connection
        if master_conn.poll():
            trainer_signal = master_conn.recv()
            # Check if it is time for testing (starts at 0%)
            if trainer_signal == "TEST":
                testResults.append(testingProcedure(path, parameters, packageName, parameters.DUR_TRAIN_TEST_NUM)[0])
                exportTestResults(testResults, path, parameters, testInterval)
            # Create or re-initialize worker pool every 'n' amount of training steps
            if trainer_signal == "RELOAD_COLLECTOR":
                terminateExperienceCollectors(collectors)
                collectors = startExperienceCollectors(parameters, expReplayer, loadedModelName, model_in_subfolder,
                                                       loadModel, path)
            # Check if 1% of training time has elapsed
            if trainer_signal == "PRINT_TRAIN_PROGRESS":
                currentPart += smallPart
                percentPrint = currentPart / parameters.MAX_TRAINING_STEPS * 100
                print("Trained: ", int(percentPrint), "%")
            if trainer_signal == "DONE":
                terminateExperienceCollectors()
                trainer.join(timeout=0.001)
                break
    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Training done.\n")
    # Testing for when training time == 100%
    testResults.append(testingProcedure(path, parameters, packageName, parameters.DUR_TRAIN_TEST_NUM)[0])
    exportTestResults(testResults, path, parameters, testInterval)


def run():
    guiEnabled = int(input("Enable GUI?: (1 == yes)\n"))
    guiEnabled = (guiEnabled == 1)
    viewEnabled = False
    if guiEnabled:
        viewEnabled = int(input("Display view?: (1 == yes)\n"))
        viewEnabled = (viewEnabled == 1)

    modelName = None
    modelPath = None
    loadedModelName = None
    algorithm = None
    packageName = None
    parameters = None
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
        if packageName is not None:
            parameters = importlib.import_module('.networkParameters', package=packageName)
            algorithm = algorithmNameToNumber(parameters.ALGORITHM)

    if not loadModel:
        packageName = "model"
        parameters = importlib.import_module('.networkParameters', package=packageName)

        # If parameter file has no algorithm, input and write to parameter file
        if parameters.ALGORITHM == "None":
            algorithm = int(input("What learning algorithm do you want to use?\n" + \
                                  "'Q-Learning' == 0, 'Actor-Critic' == 2,\n"))
            modifyParameterValue([["ALGORITHM", algorithmNumberToName(algorithm), checkValidParameter("ALGORITHM")]], modelPath)

    tweaking = int(input("Do you want to tweak parameters? (1 == yes)\n"))
    tweakedTotal = []
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

    print("Created new path: " + modelPath)

    if tweakedTotal:
        modifyParameterValue(tweakedTotal, modelPath)

    # Initialize network while number of humans is determined
    print("\nInitializing network...\n")
    p = Process(target=createNetwork, args=(parameters, modelPath))
    p.start()

    # Determine number of humans
    numberOfHumans = 0
    if guiEnabled and viewEnabled:
        numberOfHumans = int(input("Please enter the number of human players: (" + str(MAXHUMANPLAYERS) + " max)\n"))

    if guiEnabled and viewEnabled and not numberOfHumans > 0:
        spectate = int(input("Do want to spectate an individual bot's FoV? (1 = yes)\n")) == 1

    # End network init parallel process
    p.join()

    startTime = time.time()
    if numberOfHumans == 0:
        trainingProcedure(parameters, loadedModelName, model_in_subfolder, loadModel, modelPath, packageName)
    else:
        pass
    print("\nTraining proceedure time elapsed: " + str(time.time()-startTime))
    if parameters.ENABLE_TRAINING:
        runFinalTests(modelPath, parameters, packageName)
        quit()
        model.save(True)
        model.saveModels()
        if model_in_subfolder:
            print(os.path.join(modelPath))
            createCombinedModelGraphs(os.path.join(modelPath))

        print("Total average time per update: ", round(numpy.mean(model.timings), 5))

        for bot_idx, bot in enumerate([bot for bot in model.getBots() if bot.getType() == "NN"]):
            player = bot.getPlayer()
            print("")
            print("Network parameters for ", player, ":")
            attributes = dir(parameters)
            for attribute in attributes:
                if not attribute.startswith('__'):
                    print(attribute, " = ", getattr(parameters, attribute))
            print("")
            print("Mass Info for ", player, ":")
            massListPath = model.getPath() + "/data/" +  model.getDataFiles()["NN" + str(bot_idx) + "_mass"]
            with open(massListPath, 'r') as f:
                massList = list(map(float, f))
            mean = numpy.mean(massList)
            median = numpy.median(massList)
            variance = numpy.std(massList)
            print("Median = ", median, " Mean = ", mean, " Std = ", variance)
            print("")

if __name__ == '__main__':
    run()
