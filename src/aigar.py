import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #This suppresses tensorflow AVX warnings
import importlib
import importlib.util
import shutil
import psutil
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
import multiprocessing as mp
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


def elapsedTimeText(secondsElapsed):
    seconds_print = secondsElapsed % 60
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
    network = Network(parameters)
    network.saveModel(path)


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
    testParameters.GATHER_EXP = False
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
def performTest(path, specialParams):
    num_cores = mp.cpu_count()
    if num_cores > 1:
        # TODO: Make tests more peregrine-efficient by only using cores which collectors aren't using
        os.sched_setaffinity(0, range(1, num_cores)) # Core #1 is reserved for trainer process
    testParams = createTestParams(*specialParams)
    testModel = Model(False, False, testParams)
    createModelPlayers(testParams, testModel, path)
    testModel.initialize()
    for step in range(testParams.RESET_LIMIT):
        testModel.update()

    bots = testModel.getBots()
    massOverTime = [bot.getMassOverTime() for bot in bots]
    meanMass = numpy.mean([numpy.mean(botMass) for botMass in massOverTime])
    maxMeanMass = numpy.max(meanMass)
    maxMass = numpy.max([numpy.max(botMass) for botMass in massOverTime])
    varianceMass = numpy.mean(numpy.var(massOverTime))


    return [massOverTime, meanMass, maxMass, os.getpid()]


# Test the model for 'n' amount of episodes for the given type of test. This is done in parallel uses a pool of workers.
# The test results from all tests are put together into 1 structure to then be used for averaging and plotting.
def testModel(path, testType, plotName, specialParams, n_tests, testName):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Testing", testType, "...\n")

    start = time.time()
    # Parallel testing
    pool = mp.Pool(n_tests)
    print("Initializing " + str(n_tests) + " testers..." )
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    testResults = pool.starmap(performTest, [(path, specialParams) for process in range(n_tests)])
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
    evals["meanScore"] = numpy.mean(meanMasses)
    evals["stdMean"] = numpy.std(meanMasses)
    evals["meanMaxScore"] = numpy.mean(maxMasses)
    evals["stdMax"] = numpy.std(maxMasses)
    evals["maxScore"] = numpy.max(maxMasses)
    # return (name, maxScore, meanScore, stdMean, meanMaxScore, stdMax), masses
    return evals, masses


def testingProcedure(path, parameters, packageName, testName, n_tests, testResults=None, donePipe=None):
    # TODO: Make sure Actor critic noise is set to 0 while testing
    # TODO: Perform all test kinds simultaneously
    testEvals = {}
    masses = {}
    testParams = [packageName]
    testEvals["current"], masses["current"] = testModel(path, "test", "Test", testParams, n_tests, testName)

    pelletTestParams = [packageName, False]
    testEvals["pellet"], masses["pellet"] = testModel(path, "pellet", "Pellet_Collection", pelletTestParams,
                                                      n_tests, testName)
    if parameters.MULTIPLE_BOTS_PRESENT:
        greedyTestParams = (packageName, False, 1, 1)
        testEvals["vsGreedy"], masses["vsGreedy"] = testModel(path, "vsGreedy", "Vs_Greedy", greedyTestParams,
                                                              n_tests, testName)
    if parameters.VIRUS_SPAWN:
        virusTestParams = (packageName, True)
        testEvals["virus"], masses["virus"] = testModel(path, "pellet_with_virus", "Pellet_Collection_with_Viruses",
                                                        virusTestParams, n_tests, testName)
        if parameters.MULTIPLE_BOTS_PRESENT:
            virusGreedyTestParams = (packageName, True, 1, 1)
            testEvals["virusGreedy"], masses["virusGreedy"] = testModel(path, "vsGreedy_with_virus", "Vs_Greedy_with_Viruses",
                                                                        virusGreedyTestParams, n_tests, testName)
    if __debug__:
        print("Tester process sending data...")
    # This is for the case that this function is called without the need for parallelism (i.e. Final tests)
    if testResults is None:
        return testEvals, masses
    # Do this if testing procedure is being done in parallel
    else:
        # Notify master that testing is done
        donePipe.send("Done")
        # Append testEvals dictionary to list manager
        testResults.append(testEvals)


def runFinalTests(path, parameters, packageName):
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Performing final tests...\n")
    evals, masses = testingProcedure(path, parameters, packageName, "Final", parameters.DUR_TRAIN_TEST_NUM)
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
    print("\nFinal testing completed.")
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


def performModelSteps(parameters, experience_queue, processNum, model_in_subfolder, loadModel, modelPath, pauseSignal):
    num_cores = mp.cpu_count()
    if num_cores > 1:
        os.sched_setaffinity(0, range(1, num_cores)) # Core #1 is reserved for trainer process
    # Create game instance
    model = Model(False, False, parameters)
    createModelPlayers(parameters, model, modelPath)
    # TODO: Is this function call needed?
    setSeedAccordingToFolderNumber(model_in_subfolder, loadModel, modelPath, False)
    model.initialize()
    exp_req_for_put = parameters.COLLECTOR_QUEUE_PUT_EXPS * (parameters.FRAME_SKIP_RATE + 1) + 1
    # Run game until terminated
    while True:
        for step in range(parameters.RESET_LIMIT):
            if pauseSignal.value:
                if __debug__:
                    print("Collector #" + str(processNum) + " waiting for tester to finish...        <------- WAIT")
                while pauseSignal.value:
                    sleep(0.01)
                if __debug__:
                    print("Collector #" + str(processNum) + " resuming.")
            model.update()
            # After a bot has 'n' amount of experiences, send them to trainer.
            # print(len([bot.getExperiences() for bot in model.getBots()][0]))
            if (step + 1) %  exp_req_for_put == 0:
                all_experienceLists = [bot.getExperiences() for bot in model.getBots()]
                if __debug__:
                    shape = np.shape(all_experienceLists)
                    print("Collector #" + str(processNum) + " is adding to experience queue " +
                          str(shape[0]) + " lists of " + str(shape[1]) + " exps each.")
                for experienceList in all_experienceLists:
                    experience_queue.put(experienceList)
                model.resetBots()

        model.resetModel()


def startExperienceCollectors(parameters, experience_queue, loadedModelName, model_in_subfolder, loadModel, path, pauseSignal):
    startTime = time.time()
    print("*********************************************************************")
    print("Initializing collectors...")
    numWorkers = parameters.NUM_COLLECTORS
    if numWorkers <= 0:
        print("Number of concurrent games must be a positive integer.")
        quit()
    collectors = []
    for processNum in range(numWorkers):
        p = mp.Process(target=performModelSteps, args=(parameters, experience_queue, processNum, model_in_subfolder,
                                                       loadModel, path, pauseSignal))
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
    learningAlg.initializeNetwork(path)
    return learningAlg


def train(parameters, expReplayer, learningAlg, step):
    # TODO: Use GPU
    batch = expReplayer.sample(parameters.MEMORY_BATCH_LEN)
    idxs, priorities, updated_actions = learningAlg.learn(batch, step)
    if parameters.PRIORITIZED_EXP_REPLAY_ENABLED:
        expReplayer.update_priorities(idxs, numpy.abs(priorities) + 1e-4)
        if parameters.OCACLA_REPLACE_TRANSITIONS:
            if updated_actions is not None:
                expReplayer.update_dones(idxs, updated_actions)
            else:
                print("Updated actions is None!")


# Train for 'MAX_TRAINING_STEPS'. Meanwhile send signals back to master process to notify of training process.
def trainOnExperiences(parameters, experience_queue, path, queue, tr_2_m_waitPipe):
    # Increase priority of this process
    num_cores = mp.cpu_count()
    if num_cores > 1:
        os.sched_setaffinity(0, {0})  # Core #0 is reserved for trainer process
    p = psutil.Process()
    p.nice(0)
    learningAlg = createLearner(parameters, path)
    if parameters.PRIORITIZED_EXP_REPLAY_ENABLED:
        expReplayer = PrioritizedReplayBuffer(parameters.MEMORY_CAPACITY, parameters.MEMORY_ALPHA, parameters.MEMORY_BETA)
    else:
        expReplayer = ReplayBuffer(parameters.MEMORY_CAPACITY)
    # TODO: Uncomment for Anton's LSTM expReplay stuff
    # expReplayer = ExpReplay(parameters)

    # Collect enough experiences before training
    print("\n******************************************************************")
    print("Beginning initial experience collection...\n")
    collectionTime = time.time()
    while len(expReplayer) < parameters.NUM_EXPS_BEFORE_TRAIN:
        for experience in experience_queue.get():
            expReplayer.add(*experience)
        print("Buffer size: " + str(len(expReplayer)) + " | " + str(parameters.NUM_EXPS_BEFORE_TRAIN))
    # TODO: Start with buffer completely full?
    # TODO: can experiences be added in batch in Prioritized Replay Buffer?
    print("Initial experience collection completed.")
    print("Current replay buffer size:   ", len(expReplayer))
    print("Collection time elapsed:      " + str.format('{0:.3f}', time.time() - collectionTime))
    print("\nBeggining to train...")
    print("////////////////////////////////////////////////////////////////////\n")
    smallPart = max(int(parameters.MAX_TRAINING_STEPS / 100), 1)  # Get int value closest to to 1% of training time
    testInterval = smallPart * parameters.TRAIN_PERCENT_TEST_INTERVAL
    coll_stepChunk = parameters.COLLECTOR_UPDATE_STEPS
    network_saveSteps = parameters.NETWORK_SAVE_PERCENT_STEPS /100 * parameters.MAX_TRAINING_STEPS
    network_copySteps = parameters.NETWORK_COPY_PERCENT_STEPS /100 * parameters.MAX_TRAINING_STEPS
    targNet_stepChunk = parameters.TARGET_NETWORK_STEPS
    printSteps = 100
    for step in range(parameters.MAX_TRAINING_STEPS):
        if __debug__:
            if step != 0 and step % printSteps == 0:
                coll_stepsLeft = coll_stepChunk - (step % coll_stepChunk)
                test_stepsLeft = testInterval - (step % testInterval)
                targNet_stepsLeft = targNet_stepChunk - (step % targNet_stepChunk)
                save_stepsLeft = network_saveSteps - (step % network_saveSteps)
                elapsedTime = time.time() - timeStep
                print("____________________________________________________________________\n" +
                      "Current replay buffer size:                " + str(len(expReplayer)) + " | Total: " + str(parameters.MEMORY_CAPACITY) + "\n" +
                      "Steps before collector network update:      " + str(coll_stepsLeft) + " | Total: " + str(coll_stepChunk) + "\n" +
                      "Steps before next test:                     " + str(test_stepsLeft) + " | Total: " + str(testInterval) + "\n" +
                      "Steps before target network update:         " + str(targNet_stepsLeft) + " | Total: " + str(targNet_stepChunk) + "\n" +
                      "Steps before saving network:                " + str(save_stepsLeft) + " | Total: " + str(network_saveSteps) + "\n" +
                      "Total steps remaining:                      " + str(parameters.MAX_TRAINING_STEPS-step) + " | Total: " + str(parameters.MAX_TRAINING_STEPS) + "\n"
                      "Time elapsed during last " + str(printSteps) + " train steps:  " + str.format('{0:.3f}', elapsedTime) + "s   (" +
                      str.format('{0:.3f}', elapsedTime/printSteps) + "s/step)\n" +
                      "¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")
            timeStep = time.time()
        # Check if we should print the training progress percentage
        if step != 0 and step % smallPart == 0:
            if __debug__:
                print("Trainer sending signal: 'PRINT_TRAIN_PROGRESS'")
            queue.put("PRINT_TRAIN_PROGRESS")
        # Check if testing should happen
        if parameters.ENABLE_TESTING and step % testInterval == 0:
            if __debug__:
                print("Trainer sending signal: 'TEST'")
            queue.put("TEST")
            # If a tester is active before a new one is created, wait for
            if not tr_2_m_waitPipe.poll() and step != 0:
                if __debug__:
                    print("Trainer receiving wait signal...")
                tr_2_m_waitPipe.recv()
                if __debug__:
                    print("Trainer waiting for previous tester to finish...    <------- WAIT")
                tr_2_m_waitPipe.recv()
                if __debug__:
                    print("Trainer resuming.")
        # Check if copy of network weights should be saved
        if step % network_saveSteps == 0:
            # TODO: Update path name to contain number of steps during training (would allow training to resume after being stopped)
            if __debug__:
                print("Saving network...")
            # Update model.h5 network file
            learningAlg.save(path)
            if step % network_copySteps == 0:
                if __debug__:
                    print("Copying network to new file...")
                # Save this instance of the network with number of trained steps in the name
                learningAlg.save(path, str(step) + "_")
        # Check if collectors should have their networks reloaded
        if step != 0 and step % coll_stepChunk == 0:
            if __debug__:
                print("Trainer sending signal: 'RELOAD_COLLECTOR'")
            queue.put("RELOAD_COLLECTOR")
        # Get collector experiences from mp.Queue and add them to buffer
        while not experience_queue.empty():
            queueTimer = time.time()
            for experience in experience_queue.get():
                expReplayer.add(*experience)
            if __debug__:
                print("Trainer time taken to get experiences and add them to expReplay: " + str.format('{0:.3f}', time.time()-queueTimer))
        # Train 1 step
        train(parameters, expReplayer, learningAlg, step)

    # Signal that training has finished
    if __debug__:
        print("Trainer sending signal: 'DONE'")
    queue.put("DONE")
    learningAlg.save(path, str(parameters.MAX_TRAINING_STEPS) + "_")


# Create the asynchronous training procedure.
# The experience replay buffer is created as a multiprocessing.mp.Manager. This manager takes form as the expReplay class
# and is shared across all subprocesses. In order for subprocesses to access methods of the class, a Proxy manager needs
# to be made.
# The replay_buffer is first filled with enough experiences to begin training. Then, the training happens asynchronously
# from the experience collection. 'N' amount of training processes are initialized and train while 'X' amount of
# subprocesses collect experiences. After a given amount of training steps, collector subprocesses are killed and
# re-initialized with a more up-to-date version of the network.
# Every a certain amount of training, testing is done. This also happens with the training status being printed.
def trainingProcedure(parameters, loadedModelName, model_in_subfolder, loadModel, path, packageName, startTime):
    # Perform simultaneous experience collection and training
    # testerActive = False
    tr_2_m_waitPipe = m_2_tr_waitPipe = None
    c_waitSignal = False
    te_2_m_donePipe = m_2_te_donePipe = None
    if parameters.ENABLE_TESTING:
        # testerActive = mp.Value('b', False)
        tr_2_m_waitPipe, m_2_tr_waitPipe = mp.Pipe() # Trainer to master pipe
        c_waitSignal = mp.Value('b', False) # Collector to master pipe
        te_2_m_donePipe, m_2_te_donePipe = mp.Pipe() # Tester to master pipe
        testResults = mp.Manager().list()
    smallPart = max(int(parameters.MAX_TRAINING_STEPS / 100), 1)  # Get int value closest to to 1% of training time
    currentPart = 0
    testInterval = smallPart * parameters.TRAIN_PERCENT_TEST_INTERVAL
    testsPerformed = 0
    experience_queue = mp.Queue()
    collectors = startExperienceCollectors(parameters, experience_queue, loadedModelName, model_in_subfolder,
                                           loadModel, path, c_waitSignal)
    # Create training process and communication pipe
    trainer_master_queue = mp.Queue()
    # TODO: Create multiple learners
    trainer = mp.Process(target=trainOnExperiences,
                      args=(parameters, experience_queue, path, trainer_master_queue, tr_2_m_waitPipe))
    trainer.start()
    num_cores = mp.cpu_count()
    if num_cores > 1:
        os.sched_setaffinity(0, range(1,num_cores))  # Core #0 is reserved for trainer process
    while True:
        trainer_signal = trainer_master_queue.get()
        # Check if it is time for testing (starts at 0%)
        if trainer_signal == "TEST":
            if __debug__:
                print("Master received signal: 'TEST'")
            if testsPerformed != 0 and not m_2_te_donePipe.poll():
                print("Master waiting for previous tester to finish...     <------- WAIT")
                m_2_tr_waitPipe.send("WAIT")
                c_waitSignal.value = True
                m_2_te_donePipe.recv() # Master will wait until tester sends Done signal
                tester.join()
                if __debug__:
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print("Tester #" + str(testsPerformed) + " (" + testName + ") was joined by Master.")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                m_2_tr_waitPipe.send("RESUME")
                c_waitSignal.value = False
                print("Tester done. Resuming progress.\n")

            testName = str(testsPerformed*parameters.TRAIN_PERCENT_TEST_INTERVAL) + "%"
            # Start tester_process in order for testing to happen in parallel. Tester automatically appends testResults
            # to testResults list manager within process and notifies Master process of being done through the tester queue.
            tester = mp.Process(target=testingProcedure, args=(path, parameters, packageName, testName,
                                                            parameters.DUR_TRAIN_TEST_NUM, testResults, te_2_m_donePipe))
            tester.start()
            testsPerformed += 1
        # Create or re-initialize worker pool every 'n' amount of training steps
        if trainer_signal == "RELOAD_COLLECTOR":
            if __debug__:
                print("Master received signal: 'RELOAD_COLLECTOR'")
            terminateExperienceCollectors(collectors)
            collectors = startExperienceCollectors(parameters, experience_queue, loadedModelName, model_in_subfolder,
                                                   loadModel, path, c_waitSignal)
            if __debug__:
                print("¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤")
                children = mp.active_children()
                print("Number of alive child processes:    " + str(len(children)))
                for p in children:
                    print(p)
                print("¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤")
        # Check if 1% of training time has elapsed
        if trainer_signal == "PRINT_TRAIN_PROGRESS":
            if __debug__:
                print("Master received signal: 'PRINT_TRAIN_PROGRESS'")
            currentPart = printTrainProgress(parameters, currentPart, startTime)
        # When trainer signals training is 'DONE', terminate child processes (collectors and trainer) and break
        if trainer_signal == "DONE":
            if __debug__:
                print("Master received signal: 'DONE'")
            terminateExperienceCollectors(collectors)
            trainer.join(timeout=0.001)
            _ = printTrainProgress(parameters, currentPart, startTime)
            break

    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Training done.\n")
    # Testing for when training time == 100%
    if parameters.ENABLE_TESTING:
        # if not m_2_te_donePipe.poll():
        #     print("Master waiting for previous tester to finish...     <------- WAIT")
        #     m_2_tr_waitPipe.send("WAIT")
        tester.join()
        testName = str(testsPerformed * parameters.TRAIN_PERCENT_TEST_INTERVAL) + "%"
        tester = mp.Process(target=testingProcedure, args=(path, parameters, packageName, testName,
                                                                parameters.DUR_TRAIN_TEST_NUM, testResults, te_2_m_donePipe))
        tester.start()
        testsPerformed += 1
        if __debug__:
            print("Master awaiting final tester's data...")
        tester.join()
        if __debug__:
            print("Received final tester's data.")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Tester #" + str(testsPerformed) + " (" + testName + ") was joined by Master.")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        _ = printTrainProgress(parameters, currentPart, startTime)
        exportTestResults(testResults, path, parameters, testInterval)


def run():
    guiEnabled = int(input("Enable GUI?: (1 == yes)\n"))
    guiEnabled = (guiEnabled == 1)
    viewEnabled = False
    if guiEnabled:
        viewEnabled = int(input("Display view?: (1 == yes)\n"))
        viewEnabled = (viewEnabled == 1)

    tweakedTotal = []
    modelName = None
    modelPath = None
    loadedModelName = None
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

    if not loadModel:
        packageName = "model"
        parameters = importlib.import_module('.networkParameters', package=packageName)
    # TODO: Check if algorithm loads correctly ( both when starting fresh train and when loading pretrained model)
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

    print("Created new path: " + modelPath)

    if tweakedTotal:
        modifyParameterValue(tweakedTotal, modelPath)

    # Initialize network while number of humans is determined
    print("\nInitializing network...\n")
    p = mp.Process(target=createNetwork, args=(parameters, modelPath))
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
        trainingProcedure(parameters, loadedModelName, model_in_subfolder, loadModel, modelPath, packageName, startTime)
    else:
        pass
    if parameters.ENABLE_TRAINING:
        runFinalTests(modelPath, parameters, packageName)
        modelPath = finalPathName(parameters, modelPath)
        print("--------")
        print("Total time elapsed:               " + elapsedTimeText(int(time.time()- startTime)))
        print("Average time per update:    " +
              str.format('{0:.3f}', (time.time()-startTime) / parameters.MAX_TRAINING_STEPS) + " seconds")
        print("(This includes possible testing waiting time)")
        print("--------")
        if model_in_subfolder:
            print(os.path.join(modelPath))
            createCombinedModelGraphs(os.path.join(modelPath))

if __name__ == '__main__':
    run()
