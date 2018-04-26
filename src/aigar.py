import matplotlib
import sys
import importlib
import numpy

from keras.models import load_model

from controller.controller import Controller
from model.model import *
from model.network import *
from model.qLearning import *
from model.nsSarsa import *
from model.expectedSarsa import *
from model.treeBackup import *
from model.parameters import *
from model.actorCritic import *
from view.startScreen import StartScreen
from view.view import View

import numpy as np
import tensorflow as tf
import random as rn


def fix_seeds():
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

    import os
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    from keras import backend as K

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

def algorithmNumberToName(val):
    if val == 0:
        return "Q-Learning"
    elif val == 1:
        return "n-step Sarsa"
    elif val == 2:
        return "CACLA"
    elif val == 3:
        return "Discrete ACLA"
    elif val == 4:
        return "Expected Sarsa"
    elif val == 5:
        return "Tree Backup"
    else:
        print("Wrong algorithm selected...")
        quit()

def algorithmNameToNumber(name):
    if name == "Q-learning":
        return 0
    elif name == "n-step Sarsa":
        return 1
    elif name == "ACLA":
        return 2
    elif name == "Discrete ACLA":
        return 3
    elif name == "Expected Sarsa":
        return 4
    elif name == "Tree Backup":
        return 5
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
    print("Parameter with name " + tweakedParameter + "not found.")
    quit()


def modifyParameterValue(val, model, lineNumber):
    name_of_file = model.getPath() + "networkParameters.py"
    lines = open(name_of_file, 'r').readlines()
    text = ""
    for char in lines[lineNumber]:
        text += char
        if char == "=":
            break
    text += " " + str(val) + "\n"
    lines[lineNumber] = text
    out = open(name_of_file, 'w')
    out.writelines(lines)
    out.close()


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
        return int(SCREEN_WIDTH * humansNr + humansNr -1), int(SCREEN_HEIGHT)
    if humansNr == 3:
        return int(SCREEN_WIDTH * humansNr * 2/3) + humansNr -1, int(SCREEN_HEIGHT * 2/3)

    return SCREEN_WIDTH, SCREEN_HEIGHT


def createHumans(numberOfHumans, model1):
    for i in range(numberOfHumans):
        name = input("Player" + str(i + 1) + " name:\n")
        model1.createHuman(name)


def createBots(number, model, type, parameters, algorithm = None, modelName = None):
    learningAlg = None
    if type == "NN":
        Bot.num_NNbots = number

        network = Network(enableTrainMode, modelName)
        for i in range(number):
            # Create algorithm instance
            #Discrete algorithms
            if algorithm == 0:
                learningAlg = QLearn(numberOfNNBots, numberOfHumans, network, parameters)
            elif algorithm == 1:
                learningAlg = nsSarsa(numberOfNNBots, numberOfHumans, network, parameters)
            elif algorithm == 4:
                learningAlg = ExpectedSarsa(numberOfNNBots, numberOfHumans, network, parameters)
            elif algorithm == 5:
                learningAlg = TreeBackup(numberOfNNBots, numberOfHumans, network, parameters)

            #AC algorithms
            elif algorithm == 2:
                learningAlg = ActorCritic(parameters, numberOfNNBots, False, modelName)
            elif algorithm == 3:
                learningAlg = ActorCritic(parameters, numberOfNNBots, True, modelName)
            else:
                print("Please enter a valid algorithm.\n")
                quit()
            model.createBot(type, learningAlg, parameters, modelName)
    elif type == "Greedy":
        Bot.num_Greedybots = number
        for i in range(number):
            model.createBot(type)


if __name__ == '__main__':
    # This is used in case we want to use a freezing program to create an .exe
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    # fix_seeds()

    guiEnabled = int(input("Enable GUI?: (1 == yes)\n"))
    guiEnabled = (guiEnabled == 1)
    viewEnabled = False
    if guiEnabled:
        viewEnabled = int(input("Display view?: (1 == yes)\n"))
        viewEnabled = (viewEnabled == 1)
    else:
        maxSteps = int(input("For how many steps do you want to train?\n"))
    virusEnabled = int(input("Viruses enabled? (1==True)\n")) == 1
    resetSteps = int(input("Reset model after X steps (X==0 means no reset)\n"))
    model = Model(guiEnabled, viewEnabled, virusEnabled, resetSteps)

    numberOfGreedyBots = int(input("Please enter the number of Greedy bots:\n"))
    numberOfBots = numberOfGreedyBots
    if fitsLimitations(numberOfBots, MAXBOTS):
        createBots(numberOfGreedyBots, model, "Greedy", None)

    numberOfNNBots = int(input("Please enter the number of NN bots:\n"))
    numberOfBots += numberOfNNBots

    numberOfHumans = 0
    mouseEnabled = True
    humanTraining = False
    if guiEnabled and viewEnabled:
        numberOfHumans = int(input("Please enter the number of human players: (" + str(MAXHUMANPLAYERS) + " max)\n"))
        if fitsLimitations(numberOfHumans, MAXHUMANPLAYERS):
            createHumans(numberOfHumans, model)
            if numberOfHumans <= 2 and numberOfHumans > 0:
                humanTraining = int(input("Do you want to train the network using human input? (1 == yes)\n"))
                mouseEnabled = not humanTraining
            if numberOfHumans > 0 and not humanTraining:
                mouseEnabled = int(input("Do you want control Player1 using the mouse? (1 == yes)\n"))
        if numberOfBots + numberOfHumans == 0:
            modelMustHavePlayers()

        if not model.hasHuman():
            spectate = int(input("Do want to spectate an individual bot's FoV? (1 = yes)\n"))
            if spectate == 1:
                model.addPlayerSpectator()

    if fitsLimitations(numberOfBots, MAXBOTS) and (numberOfNNBots > 0 or humanTraining):
        modelName = None
        algorithm = None
        learningAlg = None
        loadModel = int(input("Do you want to load a model? (1 == yes) (2=load model from last autosave)\n"))
        if loadModel == 1:
            while modelName == None:
                modelName = input("Enter the model name (name of directory in savedModels): ")
                path = "savedModels/" + modelName
                if not os.path.exists(path):
                    print("Invalid model name, no model found under ", path)
                    modelName = None

        if loadModel == 2:
            modelName = "mostRecentAutosave"
        if modelName is not None:
            packageName = "savedModels." + modelName
            parameters = importlib.import_module('.networkParameters', package=packageName)
            algorithm = algorithmNameToNumber(parameters.ALGORITHM)
        else:
            parameters = importlib.import_module('.networkParameters', package="model")

        if loadModel == 0:
            model.initModelFolder()

            algorithm = int(input("What learning algorithm do you want to use?\n" + \
            "'Q-Learning' == 0, 'n-step Sarsa' == 1, 'CACLA' == 2,\n" + \
            "'Discrete ACLA' == 3, 'Tree Backup' == 4, 'Expected Sarsa' == 5\n"))
            tweaking = int(input("Do you want to tweak parameters? (1 == yes)\n"))
            if tweaking == 1:
                tweaked = 0
                while True:
                    tweakedParameter = str(input("Enter name of parameter to be tweaked:\n"))
                    paramLineNumber = checkValidParameter(tweakedParameter)
                    if paramLineNumber is not None:
                        paramValue = str(input("Enter parameter value:\n"))
                        modifyParameterValue(paramValue, model, paramLineNumber)
                        tweaked += 1
                    if 1 != int(input("Tweak another parameter? (1 == yes)\n")):
                        break

        enableTrainMode = humanTraining if humanTraining != None else False
        if not humanTraining:
            enableTrainMode = int(input("Do you want to train the network?: (1 == yes)\n"))
        model.setTrainingEnabled(enableTrainMode == 1)
        Bot.init_exp_replayer(parameters)
        createBots(numberOfNNBots, model, "NN", parameters, algorithm, modelName)
    if numberOfNNBots == 0:
         model.setTrainingEnabled(False)


    if numberOfBots == 0 and not viewEnabled:
        modelMustHavePlayers()

    model.initialize()

    screenWidth, screenHeight = defineScreenSize(numberOfHumans)
    model.setScreenSize(screenWidth, screenHeight)
    startScreen = StartScreen(model)

    if guiEnabled:
        view = View(model, screenWidth, screenHeight)
        controller = Controller(model, viewEnabled, view, mouseEnabled)
        view.draw()
        while controller.running:
            controller.process_input()
            model.update()
    else:
        endEpsilon = parameters.EPSILON
        startEpsilon = 1
        startEpsilon = 1
        smallPart = int(maxSteps / 200)
        for step in range(maxSteps):
            model.update()
            if step < maxSteps / 2:
                epsilon = startEpsilon - (1 - endEpsilon) * step / (maxSteps / 4)
            else:
                epsilon = endEpsilon
            model.setEpsilon(epsilon)
            if step % smallPart == 0 and step != 0:
                print("Trained: ", round(step / maxSteps * 100, 1), "%")

    if model.getTrainingEnabled():
        model.save(True)
        for bot in model.bots:
            player = bot.getPlayer()
            print("")
            print("Network parameters for ", player, ":")
            attributes = dir(parameters)
            for attribute in attributes:
                if not attribute.startswith('__'):
                    print(attribute, " = ", getattr(parameters, attribute))
            print("")
            print("Mass Info for ", player, ":")
            masses = bot.getMassOverTime()
            mean = numpy.mean(masses)
            median = numpy.median(masses)
            variance = numpy.std(masses)
            print("Median = ", median, " Mean = ", mean, " Std = ", variance)
            print("")




