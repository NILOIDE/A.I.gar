import numpy
from .cell import Cell
from .parameters import *
from .spatialHashTable import spatialHashTable


# The Field class is the main field on which cells of all sizes will move
# Its size depends on how many players are in the game
# It always contains a certain number of viruses and collectibles and regulates their number and spawnings


class Field(object):
    def __init__(self):
        self.debug = False
        self.width = 0
        self.height = 0
        self.pellets = []
        self.players = []
        self.deadPlayers = []
        self.viruses = []
        self.maxCollectibleCount = None
        self.pelletHashtable = None
        self.playerHashtable = None
        self.virusHashtable = None

    def initializePlayer(self, player):
        x = numpy.random.randint(0, self.width)
        y = numpy.random.randint(0, self.height)

        newCell = Cell(x, y, START_MASS, player)
        player.addCell(newCell)
        player.setAlive()


    def initialize(self):
        self.width = numpy.round(SIZE_INCREASE_PER_PLAYER * numpy.sqrt(len(self.players)))
        self.height = numpy.round(SIZE_INCREASE_PER_PLAYER * numpy.sqrt(len(self.players)))
        self.pelletHashtable = spatialHashTable(self.width, self.height, HASH_CELL_SIZE)
        self.playerHashtable = spatialHashTable(self.width, self.height, HASH_CELL_SIZE)
        self.virusHashtable = spatialHashTable(self.width, self.height, HASH_CELL_SIZE)
        for player in self.players:
            self.initializePlayer(player)
        self.maxCollectibleCount = self.width * self.height * MAX_COLLECTIBLE_DENSITY
        self.spawnStuff()

    def update(self):
        self.updateViruses()
        self.updatePlayers()
        self.updateHashTables()
        self.mergePlayerCells()
        self.checkOverlaps()
        self.spawnStuff()

    def updateViruses(self):
        for virus in self.viruses:
            virus.update()

    def updatePlayers(self):
        for player in self.players:
            player.update(self.width, self.height)
        self.handlePlayerCollisions()

    def handlePlayerCollisions(self):
        for player in self.players:
            self.handlePlayerCollision(player)

    def handlePlayerCollision(self, player):
        for cell in player.getCells():
            if cell.justEjected():
                continue
            for otherCell in player.getCells():
                if cell is otherCell or otherCell.justEjected() or cell.canMerge() and otherCell.canMerge():
                    continue
                distance = numpy.sqrt(cell.squaredDistance(otherCell))
                summedRadii = cell.getRadius() + otherCell.getRadius()
                if distance < summedRadii and distance != 0:
                    posDiff = cell.getPos() - otherCell.getPos()
                    scaling = (summedRadii - distance) / distance / 2
                    posDiffScaled = posDiff * scaling
                    self.adjustCellPos(cell, cell.getPos() + posDiffScaled, self.playerHashtable)
                    self.adjustCellPos(otherCell, otherCell.getPos() - posDiffScaled, self.playerHashtable)

    def updateHashTables(self):
        self.playerHashtable.clearBuckets()
        for player in self.players:
            playerCells = player.getCells()
            self.playerHashtable.insertAllObjects(playerCells)

    def mergePlayerCells(self):
        for player in self.players:
            cells = player.getMergableCells()
            if len(cells) > 1:
                cells.sort(key = lambda p: p.getMass(), reverse = True)
                for cell1 in cells:
                    if not cell1.isAlive():
                        continue
                    for cell2 in cells:
                        if (not cell2.isAlive()) or (cell2 is cell1):
                            continue
                        if cell1.overlap(cell2):
                            self.mergeCells(cell1, cell2)
                            if not cell1.isAlive():
                                break

    def checkOverlaps(self):
        self.playerPelletOverlap()
        self.playerPlayerOverlap()

    def playerPelletOverlap(self):
        for player in self.players:
            for cell in player.getCells():
                for collectible in self.pelletHashtable.getNearbyObjects(cell):
                    if cell.overlap(collectible):
                        self.eatPellet(cell, collectible)


    def playerPlayerOverlap(self):
        for player in self.players:
            for playerCell in player.getCells():
                if not playerCell.isAlive():
                    if self.debug:
                      print("Skip cell ", playerCell," because it is dead!")
                    continue
                opponentCells = self.playerHashtable.getNearbyEnemyObjects(playerCell)
                if self.debug:
                    if len(opponentCells) > 0:
                        print("\n_________")
                        print("Opponent cells of cell ", playerCell, ":")
                        for cell in opponentCells:
                            print(cell, end= " ")
                        print("\n____________\n")
                for opponentCell in opponentCells:
                        if playerCell.overlap(opponentCell):
                            if self.debug:
                                print(playerCell, " and ", opponentCell, " overlap!")
                            if playerCell.canEat(opponentCell):
                                self.eatPlayerCell(playerCell, opponentCell)
                            elif opponentCell.canEat(playerCell):
                                self.eatPlayerCell(opponentCell, playerCell)
                                break

    def spawnStuff(self):
        self.spawnPellets()
        self.spawnViruses()
        self.spawnPlayers()

    def spawnViruses(self):
        pass

    def spawnPlayers(self):
        for player in self.players:
            if len(player.getCells()) < 1:
                if self.debug:
                    print(player.getName(), " died!")
                self.initializePlayer(player)
                if self.debug:
                    print("REVIVE ", player.getName(), "!!!")

    def spawnPellets(self):
        while len(self.pellets) < self.maxCollectibleCount:
            self.spawnPellet()

    def spawnPellet(self):
        xPos = numpy.random.randint(0, self.width)
        yPos = numpy.random.randint(0, self.height)
        size = self.randomSize()
        pellet = Cell(xPos, yPos, size, None)
        self.addPellet(pellet)

    # Cell1 eats Cell2. Therefore Cell1 grows and Cell2 is deleted
    def eatPellet(self, cell, pellet):
        self.adjustCellSize(cell, pellet.getMass(), self.playerHashtable)
        self.pellets.remove(pellet)
        self.pelletHashtable.deleteObject(pellet)
        pellet.setAlive(False)

    def eatPlayerCell(self, largerCell, smallerCell):
        if self.debug:
            print(largerCell, " eats ", smallerCell, "!")
        self.adjustCellSize(largerCell, smallerCell.getMass(), self.playerHashtable)
        self.deletePlayerCell(smallerCell)

    def mergeCells(self, firstCell, secondCell):
        if firstCell.getMass() > secondCell.getMass():
            biggerCell = firstCell
            smallerCell = secondCell
        else:
            biggerCell = secondCell
            smallerCell = firstCell
        if self.debug:
            print(smallerCell, " is merged into ", biggerCell, "!")
        self.adjustCellSize(biggerCell, smallerCell.getMass(), self.playerHashtable)
        self.deletePlayerCell(smallerCell)

    def deletePlayerCell(self, playerCell):
        self.playerHashtable.deleteObject(playerCell)
        player = playerCell.getPlayer()
        player.removeCell(playerCell)


    def randomSize(self):
        maxRand = 20
        maxPelletSize = 5
        sizeRand = numpy.random.randint(0, maxRand)
        if sizeRand > (maxRand - maxPelletSize):
            return maxRand - sizeRand
        return 1

    def addPellet(self, pellet):
        self.pelletHashtable.insertObject(pellet)
        self.pellets.append(pellet)

    def adjustCellSize(self, cell, mass, hashtable):
        hashtable.deleteObject(cell)
        cell.grow(mass)
        hashtable.insertObject(cell)

    def adjustCellPos(self, cell, newPos, hashtable):
        #hashtable.deleteObject(cell)
        x = min(self.width, max(0, newPos[0]))
        y = min(self.height, max(0, newPos[1]))
        cell.setPos(x, y)
        #hashtable.insertObject(cell)

    # Setters:
    def setDebug(self, val):
        self.debug = val

    def addPlayer(self, player):
        player.setAlive()
        self.players.append(player)

    # Getters:
    def getEnemyPlayerCellsInFov(self, fovPlayer):
        fovPos = fovPlayer.getFovPos()
        fovDims = fovPlayer.getFovDims()
        cellsInFov = self.getCellsFromHashtableInFov(self.playerHashtable, fovPos, fovDims)
        opponentCellsInFov = []
        for playerCell in cellsInFov:
            # If the playerCell is an opponent Cell
            if playerCell.getName() != fovPlayer.getName() and playerCell.isInFov(fovPos, fovDims):
                opponentCellsInFov.append(playerCell)
        return opponentCellsInFov


    def getPelletsInFov(self, fovPlayer):
        fovPos = fovPlayer.getFovPos()
        fovDims = fovPlayer.getFovDims()
        cellsInFov = self.getCellsFromHashtableInFov(self.pelletHashtable, fovPos, fovDims)
        pelletsInFov = []
        for pellet in cellsInFov:
            if pellet.isInFov(fovPos, fovDims):
                pelletsInFov.append(pellet)
        return pelletsInFov

    def getCellsFromHashtableInFov(self, hashtable, fovPos, fovDims):
        fovCell = Cell(fovPos[0], fovPos[1], 1, None)
        fovCell.setRadius(fovDims[0] / 2)
        return hashtable.getNearbyObjects(fovCell)


    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getPellets(self):
        return self.pellets

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
