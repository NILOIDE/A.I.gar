import pygame
import numpy
from pygame import gfxdraw

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class RGBGenerator:
    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters

        # CNN attributes
        if self.parameters.CNN_USE_LAYER_1:
            self.length = self.parameters.CNN_SIZE_OF_INPUT_DIM_1
        elif self.parameters.CNN_USE_LAYER_2:
            self.length = self.parameters.CNN_SIZE_OF_INPUT_DIM_2
        else:
            self.length = self.parameters.CNN_SIZE_OF_INPUT_DIM_3

        self.screenDims = numpy.array([self.length, self.length])
        self.screen = pygame.Surface((self.length, self.length))

        if __debug__:
            self.screen = pygame.display.set_mode(self.screenDims)
            pygame.init()
            pygame.display.set_caption('A.I.gar')

    def drawCells(self, cells, fovPos, fovSize):
        for cell in cells:
            self.drawSingleCell(cell, fovPos, fovSize)


    def drawSingleCell(self, cell, fovPos, fovSize):
        screen = self.screen
        unscaledRad = cell.getRadius()
        unscaledPos = numpy.array(cell.getPos())
        color = cell.getColor()
        if __debug__ and cell.getPlayer():
            if cell.getPlayer().getSelected():
                color = (255, 0, 0)
            if cell.getPlayer().isExploring():
                color = (0, 255, 0)

        player = cell.getPlayer()
        rad = int(self.modelToViewScaleRadius(unscaledRad, fovSize))
        pos = self.modelToViewScaling(unscaledPos, fovPos, fovSize).astype(int)
        if rad >= 4:
            pygame.gfxdraw.filled_circle(screen, pos[0], pos[1], rad, color)
            if cell.getName() == "Virus":
                # Give Viruses a black surrounding circle
                pygame.gfxdraw.aacircle(screen, pos[0], pos[1], rad, (0,0,0))
            else:
                pygame.gfxdraw.aacircle(screen, pos[0], pos[1], rad, color)
        else:
            # Necessary to avoid that collectibles are drawn as little X's when the fov is huge
            pygame.draw.circle(screen, color, pos, rad)
        if player is not None or (__debug__ and cell.getName() == "Virus"):
            font = pygame.font.SysFont(None, int(rad / 2))
            name = font.render(cell.getName(), True, (0,0,0))
            textPos = [pos[0] - name.get_width() / 2, pos[1] - name.get_height() / 2]
            screen.blit(name, textPos)
            if __debug__:
                mass = font.render("Mass:" + str(int(cell.getMass())), True, (0, 0, 0))
                textPos = [pos[0] - mass.get_width() / 2, pos[1] - mass.get_height() / 2 + name.get_height()]
                screen.blit(mass, textPos)
                if cell.getMergeTime() > 0:
                    text = font.render(str(int(cell.getMergeTime())), True, (0, 0, 0))
                    textPos = [pos[0] - text.get_width() / 2, pos[1] - text.get_height() / 2 + name.get_height() + mass.get_height()]
                    screen.blit(text, textPos)


    def drawAllCells(self, player):

        fovPos = player.getFovPos()
        fovSize = player.getFovSize()
        pellets = self.model.getField().getPelletsInFov(fovPos, fovSize)
        blobs = self.model.getField().getBlobsInFov(fovPos, fovSize)
        viruses = self.model.getField().getVirusesInFov(fovPos, fovSize)
        playerCells = self.model.getField().getPlayerCellsInFov(fovPos, fovSize)
        allCells = pellets + blobs + viruses + playerCells
        allCells.sort(key = lambda p: p.getMass())

        self.drawCells(allCells, fovPos, fovSize)


    def draw_cnnInput(self, player):
        self.screen.fill(WHITE)
        self.drawAllCells(player)
        if __debug__:
            pygame.display.update()

    def modelToViewScaling(self, pos, fovPos, fovSize):
        adjustedPos = pos - fovPos + (fovSize / 2)
        scaledPos = adjustedPos * (self.screenDims / fovSize)
        return scaledPos


    def viewToModelScaling(self, pos, fovPos, fovSize):
        scaledPos = pos / (self.screenDims / fovSize)
        adjustedPos = scaledPos + fovPos - (fovSize / 2)
        return adjustedPos


    def modelToViewScaleRadius(self, rad, fovSize):
        return rad * (self.screenDims[0] / fovSize)

    def get_cnn_inputRGB(self, player):
        self.draw_cnnInput(player)
        imgdata = pygame.surfarray.array3d(self.screen)
        print("imgdata shape: ", numpy.shape(imgdata))
        # print(imgdata)
        return imgdata


