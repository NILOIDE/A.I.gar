import os
import importlib
import shutil
import time
import datetime
#import pyximport; pyximport.install()
from controller.controller import Controller
from model.qLearning import *
from model.actorCritic import *
from model.bot import *
from model.model import Model
import matplotlib.pyplot as plt
import pickle as pkl
import subprocess
from builtins import input
import pathos.multiprocessing as mp
from time import sleep

from view.view import View
from modelCombiner import createCombinedModelGraphs, plot

import numpy as np
import tensorflow as tf
import random as rn


def fix_seeds(seedNum):
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
    elif name == "AC":
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
        # if tweaked[i][0] == "RESET_LIMIT":
        #     model.resetLimit = int(tweaked[i][1])
        print(tweaked[i][0])
        text += " " + str(tweaked[i][1]) + "\n"
        lines[tweaked[i][2]] = text
    out = open(name_of_file, 'w')
    out.writelines(lines)
    out.close()
    # parameters = importlib.import_module('.networkParameters', package=model.getPath().replace("/", ".")[:-1])
    # model.initParameters(parameters)


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


def createHumans(numberOfHumans, model1):
    for i in range(numberOfHumans):
        name = input("Player" + str(i + 1) + " name:\n")
        model1.createHuman(name)


def createBots(number, model, botType, parameters, algorithm=None, loadModel=None):
    learningAlg = None
    loadPath = model.getPath() if loadModel else None
    if botType == "NN":
        Bot.num_NNbots = number
        networks = {}
        for i in range(number):
            # Create algorithm instance
            if algorithm == 0:
                learningAlg = QLearn(parameters)
            elif algorithm == 2:
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


def performTest(testParams, testSteps):

    testModel = Model(False, False, testParams, False)
    # pelletModel.createBot("NN", currentAlg, parameters)
    # TODO: Change to NN bots
    createBots(testParams.NUM_NN_BOTS, testModel, "Greedy", testParams,0)

    bots = testModel.getBots()
    testModel.initialize(False)
    for step in range(testSteps):
        testModel.update()

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


def testModel(n_tests, testSteps, modelPath, name, testParams=None):
    print("Testing", name, "...")
    start = time.time()

    # Serial Testing
    currentEval = []
    # for i in range(n_training):
    #     currentEval.append(performTest(testParams, testSteps))

    # Parallel testing
    pool = mp.Pool(n_tests)
    testResults = pool.starmap(performTest, [(testParams, testSteps) for process in range(n_tests)])
    print("Number of tests: ", n_tests, "Time elapsed: ", time.time() - start, "\n")
    print("------------------------------------------------------------------\n")

    masses = []
    meanMasses = []
    maxMasses = []
    for test in range(n_tests):
        masses.extend(testResults[test][0])
        meanMasses.append(testResults[test][1])
        maxMasses.append(testResults[test][2])

    meanScore = numpy.mean(meanMasses)
    stdMean = numpy.std(meanMasses)
    meanMaxScore = numpy.mean(maxMasses)
    stdMax = numpy.std(maxMasses)
    maxScore = numpy.max(maxMasses)
    # TODO: fix plotting
    # plotPath = modelPath
    # modelPath += "data/"
    # if not os.path.exists(modelPath):
    #     os.mkdir(modelPath)
    # if plotting:
    #     meanMassPerTimeStep = []
    #     for timeIdx in range(reset_time):
    #         val = 0
    #         for listIdx in range(n_training):
    #             val += masses[listIdx][timeIdx]
    #         meanVal = val / n_training
    #         meanMassPerTimeStep.append(meanVal)
    #
    #     #exportTestResults(meanMassPerTimeStep, modelPath, "Mean_Mass_" + name)
    #     labels = {"meanLabel": "Mean Mass", "sigmaLabel": '$\sigma$ range', "xLabel": "Step number",
    #               "yLabel": "Mass mean value", "title": "Mass plot test phase", "path": plotPath,
    #               "subPath": "Mean_Mass_" + name}
    #     plot(masses, reset_time, 1, labels)

    return name, maxScore, meanScore, stdMean, meanMaxScore, stdMax


def updateTestResults(testResults, modelPath, parameters):
    # currentAlg = model.getNNBot().getLearningAlg()
    # originalNoise = currentAlg.getNoise()
    # currentAlg.setNoise(0)
    #
    # originalTemp = None
    # if str(currentAlg) != "AC":
    #     originalTemp = currentAlg.getTemperature()
    #     currentAlg.setTemperature(0)

    # TODO: Perform all test kinds simultaneously
    n_tests = parameters.DUR_TRAIN_TEST_NUM

    currentEval = testModel(n_tests, parameters.RESET_LIMIT, modelPath, "test", parameters)

    pelletTestParams = modifyParameters(parameters, False)
    pelletEval = testModel(n_tests, parameters.RESET_LIMIT, modelPath, "pellet", pelletTestParams)

    vsGreedyEval = (0, 0, 0, 0)
    virusGreedyEval = (0, 0, 0, 0)
    virusEval = (0, 0, 0, 0)

    if parameters.MULTIPLE_BOTS_PRESENT:
        greedyTestParams = modifyParameters(parameters, False, 1, 1)
        vsGreedyEval = testModel(n_tests, parameters.RESET_LIMIT, modelPath, "vsGreedy", greedyTestParams)

    if parameters.VIRUS_SPAWN:
        virusTestParams = modifyParameters(parameters, True)
        virusEval = testModel(n_tests, parameters.RESET_LIMIT, modelPath, "pellet_with_virus", virusTestParams)
        if parameters.MULTIPLE_BOTS_PRESENT:
            virusGreedyTestParams = modifyParameters(parameters, True, 1, 1)
            virusGreedyEval = testModel(n_tests, parameters.RESET_LIMIT, modelPath, "vsGreedy_with_virus", virusGreedyTestParams)

    # TODO: Check if following commented noise code is needed
    # currentAlg.setNoise(originalNoise)
    #
    # if str(currentAlg) != "AC":
    #     currentAlg.setTemperature(originalTemp)

    meanScore = currentEval[2]
    stdDev = currentEval[3]
    testResults.append((meanScore, stdDev, pelletEval[2], pelletEval[3],
                        vsGreedyEval[2], vsGreedyEval[3], virusEval[2], virusEval[3], virusGreedyEval[2], virusGreedyEval[3]))
    return testResults


def modifyParameters(parameters, virus, num_nn_bots=1, num_greedy_bots=0, num_rand_bots=0):
    testParameters = parameters
    testParameters.VIRUS_SPAWN = virus
    testParameters.NUM_NN_BOTS = num_nn_bots
    testParameters.NUM_GREEDY_BOTS = num_greedy_bots
    testParameters.NUM_RAND_BOTS = num_rand_bots
    return testParameters


def exportTestResults(testResults, path, name):
    filePath = path + name + ".txt"
    with open(filePath, "a") as f:
        for val in testResults:
            # write as: "mean\n"
            line = str(val) + "\n"
            f.write(line)


def plotTesting(testResults, path, timeBetween, end, name, idxOfMean):
    x = range(0, end + timeBetween, timeBetween)
    y = [x[idxOfMean] for x in testResults]
    ysigma = [x[idxOfMean + 1] for x in testResults]

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
    title =  name + " mass over time"

    meanY = numpy.mean(y)
    ax.legend(loc='upper left')
    ax.set_ylabel(yLabel)
    ax.set_title(title + " mean value (" + str(round(meanY, 1)) + ") $\pm$ $\sigma$ interval")
    ax.grid()
    fig.savefig(path + title + ".pdf")

    plt.close()

def runTests(model, parameters):
    np.random.seed()

    print("Testing...")
    # Set Parameters:
    resetPellet = 15000
    resetGreedy = 30000
    resetVirus = 15000
    n_test_runs = 10
    trainedBot = model.getNNBot()
    trainedAlg = trainedBot.getLearningAlg()
    evaluations = []
    # Pellet testing:
    params = Params(0, False, parameters.EXPORT_POINT_AVERAGING)

    pelletModel = Model(False, False, params, False)
    pelletModel.createBot("NN", trainedAlg, parameters)
    pelletEvaluation = testModel(pelletModel, n_test_runs, resetPellet, model.getPath(), "pellet_collection")
    evaluations.append(pelletEvaluation)
    # Greedy Testing:
    if len(model.getBots()) > 1:
        greedyModel = Model(False, False, params, False)
        greedyModel.createBot("NN", trainedAlg, parameters)
        greedyModel.createBot("Greedy", None, parameters)
        greedyEvaluation = testModel(greedyModel, n_test_runs, resetGreedy, model.getPath(), "vs_1_greedy")
        evaluations.append(greedyEvaluation)
    # Virus Testing:
    if model.getVirusEnabled():
        params = Params(0, True, parameters.EXPORT_POINT_AVERAGING)
        virusModel = Model(False, False, params, False)
        virusModel.createBot("NN", trainedAlg, parameters)
        virusEvaluation = testModel(virusModel, n_test_runs, resetVirus, model.getPath(), "virus")
        evaluations.append(virusEvaluation)

    # TODO: add more test scenarios for multiple greedy bots and full model check
    print("Testing completed.")

    name_of_file = model.getPath() + "/final_results.txt"
    with open(name_of_file, "w") as file:
        data = "Avg run time(s): " + str(round(numpy.mean(model.timings), 6)) + "\n"
        data += "Number of runs per testing: " + str(n_test_runs) + "\n"
        for evaluation in evaluations:
            name = evaluation[0]
            maxScore = str(round(evaluation[1], 1))
            meanScore = str(round(evaluation[2], 1))
            stdMean = str(round(evaluation[3], 1))
            meanMaxScore = str(round(evaluation[4], 1))
            stdMax = str(round(evaluation[5], 1))
            data += name + " Highscore: " + maxScore + " Mean: " + meanScore + " StdMean: " + stdMean \
                    + " Mean_Max_Score: " + meanMaxScore + " Std_Max_Score: " + stdMax + "\n"
        file.write(data)


def modelPlayers(parameters, model, numberOfHumans, algorithm, loadModel):
    parameters = importlib.import_module('.networkParameters', package=model.getPath().replace("/", ".")[:-1])
    numberOfNNBots = parameters.NUM_NN_BOTS
    numberOfGreedyBots = parameters.NUM_GREEDY_BOTS
    numberOfBots = numberOfNNBots + numberOfGreedyBots

    if numberOfNNBots == 0:
        model.setTrainingEnabled(False)

    if numberOfBots == 0 and not model.viewEnabled:
        modelMustHavePlayers()

    createHumans(numberOfHumans, model)
    createBots(numberOfNNBots, model, "NN", parameters, algorithm, loadModel)
    createBots(numberOfGreedyBots, model, "Greedy", parameters)
    createBots(parameters.NUM_RANDOM_BOTS, model, "Random", parameters)


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

    modelPlayers(parameters, model, numberOfHumans, algorithm, loadModel)

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


def performModelSteps(parameters, enableTrainMode, loadedModelName, model_in_subfolder, loadModel, modelPath, algorithm):
    model = Model(False, False, parameters, True)
    model.setTrainingEnabled(enableTrainMode == 1)

    modelPlayers(parameters, model, 0, algorithm, loadModel)

    Bot.init_exp_replayer(parameters, loadedModelName)

    setSeedAccordingToFolderNumber(model_in_subfolder, loadModel, modelPath, enableTrainMode)

    model.initialize(loadModel)

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


def run():
    # This is used in case we want to use a freezing program to create an .exe
    #if getattr(sys, 'frozen', False):
    #    os.chdir(sys._MEIPASS)

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
            # model.setPath(modelName)

    if not loadModel:
        parameters = importlib.import_module('.networkParameters', package="model")

        algorithm = int(input("What learning algorithm do you want to use?\n" + \
                              "'Q-Learning' == 0, 'Actor-Critic' == 2,\n"))
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

    numberOfHumans = 0
    if guiEnabled and viewEnabled:
        numberOfHumans = int(input("Please enter the number of human players: (" + str(MAXHUMANPLAYERS) + " max)\n"))

    numWorkers = 1
    if numberOfHumans == 0:
        numWorkers = int(input("Number of concurrent games? (1 == yes)\n"))
        if numWorkers <= 0:
            print("Number of concurrent games must be a positive integer.")
            quit()

    enableTrainMode = int(input("Do you want to train the network?: (1 == yes)\n"))

    if guiEnabled and viewEnabled and not numberOfHumans > 0:
        spectate = int(input("Do want to spectate an individual bot's FoV? (1 = yes)\n")) == 1

    mp.Pool(numWorkers)

    testResults = []
    testResults = updateTestResults(testResults, modelPath, parameters)
    meanMassesOfTestResults = [val[0] for val in testResults]
    exportTestResults(meanMassesOfTestResults, model.getPath() + "data/", "testMassOverTime")
    meanMassesOfPelletResults = [val[2] for val in testResults]
    exportTestResults(meanMassesOfPelletResults, model.getPath() + "data/", "Pellet_CollectionMassOverTime")
    plotTesting(testResults, model.getPath(), testPercentage, maxSteps, "Test", 0)
    plotTesting(testResults, model.getPath(), testPercentage, maxSteps, "Pellet_Collection", 2)

    if parameters.MULTIPLE_BOTS_PRESENT:
        meanMassesOfGreedyResults = [val[4] for val in testResults]
        exportTestResults(meanMassesOfGreedyResults, model.getPath() + "data/", "VS_1_GreedyMassOverTime")
        plotTesting(testResults, model.getPath(), testPercentage, maxSteps, "Vs_Greedy", 4)
    if parameters.VIRUS_SPAWN:
        meanMassesOfPelletVirusResults = [val[6] for val in testResults]
        exportTestResults(meanMassesOfPelletVirusResults, model.getPath() + "data/", "Pellet_Collection_Virus_MassOverTime")
        plotTesting(testResults, model.getPath(), testPercentage, maxSteps, "Pellet_Collection_with_Viruses", 6)
        if parameters.MULTIPLE_BOTS_PRESENT:
            meanMassesOfGreedyVirusResults = [val[8] for val in testResults]
            exportTestResults(meanMassesOfGreedyVirusResults, model.getPath() + "data/", "VS_1_Greedy_Virus_MassOverTime")
            plotTesting(testResults, model.getPath(), testPercentage, maxSteps, "Vs_Greedy_with_Viruses", 8)


        print("Training done.")
        print("")

    if model.getTrainingEnabled():
        model.save(True)
        model.saveModels()
        runTests(model, parameters)
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
