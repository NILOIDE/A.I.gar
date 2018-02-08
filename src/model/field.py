from random import randint
import numpy
from .cell import Cell
from .parameters import *

# The Field class is the main field on which cells of all sizes will move
# Its size depends on how many players are in the game
# It always contains a certain number of viruses and collectibles and regulates their number and spawnings




class Field(object):
    def __init__(self):
        self.width = 0
        self.height = 0
        self.collectibles = []
        self.players = []
        self.deadPlayers = []
        self.viruses = []
        self.maxCollectibleCount = None

    def initializePlayer(self, player):
        x = randint(0, self.width)
        y = randint(0, self.height)
        newCell = Cell(x, y, START_MASS, player.getColor())
        player.addCell(newCell)

    def initialize(self):
        self.width = numpy.round(SIZE_INCREASE_PER_PLAYER * numpy.sqrt(len(self.players)))
        self.height = numpy.round(SIZE_INCREASE_PER_PLAYER * numpy.sqrt(len(self.players)))
        for player in self.players:
            self.initializePlayer(player)
        self.maxCollectibleCount = self.width * self.height * MAX_COLLECTIBLE_DENSITY

        self.spawnStuff(MAX_COLLECTIBLE_SPAWN_PER_UPDATE)

    def update(self):
        self.updateViruses()
        self.updatePlayers()
        self.checkCollisions()

        self.spawnStuff(MAX_COLLECTIBLE_SPAWN_PER_UPDATE)

    def checkCollisions(self):
        self.collectibleCollisions()
        self.playerCollisions()


    def collectibleCollisions(self):
        for player in self.players:
            for cell in player.getCells():
                for collectible in self.collectibles:
                    if cell.overlap(collectible):
                        self.eatCollectible(cell, collectible)

    def playerCollisions(self):
        for i in range(len(self.players)):
            for j in range(i, len(self.players)):
                for playerCell in self.players[i].getCells():
                    for opponentCell in self.players[j].getCells():
                        if playerCell.overlap(opponentCell):
                            if playerCell.getMass() > 1.1*opponentCell.getMass():
                                self.eatPlayerCell(playerCell, opponentCell, self.players[j])
                            if playerCell.getMass()*1.1 < opponentCell.getMass():
                                self.eatPlayerCell(opponentCell, playerCell, self.players[i])



    # Cell1 eats Cell2. Therefore Cell1 grows and Cell2 is deleted
    def eatCollectible(self, cell, collectible):
        cell.grow(collectible.getMass())
        self.collectibles.remove(collectible)

    def eatPlayerCell(self, largerCell, smallerCell, smallerPlayer):
        largerCell.grow(smallerCell.getMass())
        smallerPlayer.removeCell(smallerCell)
        if (len(smallerPlayer.getCells()) == 0):
            smallerPlayer.setDead()
            self.deadPlayers.append(smallerPlayer)


    def updateViruses(self):
        for virus in self.viruses:
            virus.update()


    def updatePlayers(self):
        for player in self.players:
            player.update(self.width, self.height)

    def spawnStuff(self, maxSpawns):
        self.spawnCollectibles(maxSpawns)
        self.spawnViruses(maxSpawns)

    def spawnCollectibles(self, maxSpawns):
        count = 0
        while len(self.collectibles) < self.maxCollectibleCount and count < maxSpawns:
            self.spawnCollectible()
            count += 1

    def spawnCollectible(self):
        xPos = randint(0, self.width)
        yPos = randint(0, self.height)
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        collectible = Cell(xPos, yPos, COLLECTIBLE_SIZE, color)
        self.addCollectible(collectible)

    def spawnViruses(self, maxSpawns):
        pass

    def removeDeadPlayer(self, player):
        self.deadPlayers.remove(player)

    def addCollectible(self, collectible):
        self.collectibles.append(collectible)

    # Setters:
    def addPlayer(self, player):
        self.players.append(player)

    # Getters:
    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getCollectibles(self):
        return self.collectibles

    def getViruses(self):
        return self.viruses

    def getPlayerCells(self):
        cells = []
        for player in self.players:
            cells += player.getCells()
        return cells

    def getDeadPlayers(self):
        return self.deadPlayers

    def getPlayers(self):
        return self.players
