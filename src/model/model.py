import time
import datetime
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
import numpy

from .bot import Bot
from .field import Field
from .parameters import *
from .player import Player
from model.rgbGenerator import *

import linecache
import os
import tracemalloc
import pickle as pkl

# Useful function that displays the top 3 lines that use the most total memory so far
def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


# The model class is the main wrapper for the game engine.
# It contains the field and the players.
# It links the actions of the players to consequences in the field and updates information.

class Model(object):
    def __init__(self, guiEnabled, viewEnabled, parameters):
        self.listeners = []
        self.viewEnabled = viewEnabled
        self.guiEnabled = guiEnabled
        self.parameters = parameters
        self.virusEnabled = parameters.VIRUS_SPAWN
        self.resetLimit = parameters.RESET_LIMIT
        self.path = None
        self.superPath = None
        self.startTime = None

        self.players = []
        self.bots = []
        self.humans = []
        self.playerSpectator = None
        self.spectatedPlayer = None
        self.field = Field(self.virusEnabled)
        self.screenWidth = None
        self.screenHeight = None
        self.counter = 0
        self.timings = []
        self.rewards = []
        self.tdErrors = []
        self.dataFiles = {}

        if __debug__:
            tracemalloc.start()

    def initParameters(self, parameters):
        self.parameters = parameters
        self.virusEnabled = parameters.VIRUS_SPAWN
        self.resetLimit = parameters.RESET_LIMIT
        # self.pointAveraging = parameters.EXPORT_POINT_AVERAGING
        self.field = Field(self.virusEnabled)


    def modifySettings(self, reset_time):
        self.resetLimit = reset_time

    def initialize(self):
        if __debug__:
            print("Initializing model...")
        self.field.initialize()
        self.resetBots()

    def resetModel(self):
        self.field.reset()
        self.counter = 0

    def update(self):
        self.counter += 1

        timeStart = time.time()
        # Get the decisions of the bots. Update the field accordingly.
        self.takeBotActions()
        self.field.update()
        # Update view if view is enabled
        if self.guiEnabled and self.viewEnabled:
            self.notify()
        # Slow down game to match FPS
        if self.humans:
            time.sleep(max( (1/FPS) - (time.time() - timeStart),0))

    def takeBotActions(self):
        for bot in self.bots:
            bot.makeMove()

    def resetBots(self):
        for bot in self.bots:
            bot.reset()

    def plotSPGTrainingCounts(self):
        for bot_idx, bot in enumerate(self.bots):
            playerName = str(bot.getPlayer())
            name = "BatchSizeOverTime" + playerName
            if bot.learningAlg is not None and str(bot.learningAlg) == "AC":
                counts = bot.learningAlg.counts
                len_counts = len(counts)
                y = [numpy.mean(counts[idx:idx + self.pointAveraging]) for idx in range(0, len_counts, self.pointAveraging)]
                timeAxis = list(range(0, len_counts, self.pointAveraging))

                plt.plot(timeAxis, y)
                plt.title("Actor Training Batch Size During Training")
                plt.xlabel("Training Steps")
                plt.ylabel("Batch Size")
                plt.savefig(self.path + name + ".pdf")
                plt.close()

                # Export counts:
                with open(self.path + "data/" + name + ".txt", "w") as f:
                    for item in counts:
                        f.write("%s\n" % item)

    def printBotMasses(self):
        for bot in self.bots:
            mass = bot.getPlayer().getTotalMass()
            print("Mass of ", bot.getPlayer(), ": ", round(mass, 1) if mass is not None else "Dead")

    def createPlayer(self, name):
        newPlayer = Player(name)
        self.addPlayer(newPlayer)
        return newPlayer

    def createBot(self, botType, learningAlg = None, parameters = None):
        name = botType + str(len(self.bots))
        newPlayer = self.createPlayer(name)
        rgbGenerator = None
        if parameters is not None:
            if parameters.CNN_REPR and parameters.CNN_P_REPR:
                rgbGenerator = RGBGenerator(self.field, parameters)
        bot = Bot(newPlayer, self.field, botType, learningAlg, parameters, rgbGenerator)
        self.addBot(bot)

    def createHuman(self, name):
        newPlayer = self.createPlayer(name)
        self.addHuman(newPlayer)

    def addPlayer(self, player):
        self.players.append(player)
        self.field.addPlayer(player)

    def addBot(self, bot):
        self.bots.append(bot)
        player = bot.getPlayer()
        if player not in self.players:
            self.addPlayer(player)

    def addHuman(self, human):
        self.humans.append(human)

    def addPlayerSpectator(self):
        self.playerSpectator = True
        self.setSpectatedPlayer(self.players[0])

    def setPath(self, path):
        self.path = path

    def setSpectatedPlayer(self, player):
        self.spectatedPlayer = player

    def setViewEnabled(self, boolean):
        self.viewEnabled = boolean

    def setScreenSize(self, width, height):
        self.screenWidth = width
        self.screenHeight = height

    # Checks:
    def hasHuman(self):
        return bool(self.humans)

    def hasPlayerSpectator(self):
        return self.playerSpectator is not None

    # Getters:
    def getNNBot(self):
        for bot in self.bots:
            if bot.getType() == "NN":
                return bot

    def getNNBots(self):
        return [bot for bot in self.bots if bot.getType() == "NN"]

    def getTopTenPlayers(self):
        players = self.getPlayers()[:]
        players.sort(key=lambda p: p.getTotalMass(), reverse=True)
        return players[0:10]


    def getHumans(self):
        return self.humans

    def getFovPos(self, humanNr):
        if self.hasHuman():
            fovPos = numpy.array(self.humans[humanNr].getFovPos())
        elif self.hasPlayerSpectator():
            fovPos = numpy.array(self.spectatedPlayer.getFovPos())
        else:
            fovPos = numpy.array([self.field.getWidth() / 2, self.field.getHeight() / 2])
        return fovPos

    def getFovSize(self, humanNr):
        if self.hasHuman():
            fovSize = self.humans[humanNr].getFovSize()
        elif self.hasPlayerSpectator():
            fovSize = self.spectatedPlayer.getFovSize()
        else:
            fovSize = self.field.getWidth()
        return fovSize

    def getField(self):
        return self.field

    def getPellets(self):
        return self.field.getPellets()

    def getViruses(self):
        return self.field.getViruses()

    def getPlayers(self):
        return self.players

    def getBots(self):
        return self.bots

    def getPlayerCells(self):
        return self.field.getPlayerCells()

    def getSpectatedPlayer(self):
        if self.hasHuman():
            return self.humans
        if self.hasPlayerSpectator():
            return self.spectatedPlayer
        return None

    def getParameters(self):
        return self.parameters

    def getVirusEnabled(self):
        return self.virusEnabled

    # MVC related method
    def set_GUI(self, value):
        self.guiEnabled = value

    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self):
        for listener in self.listeners:
            listener()
