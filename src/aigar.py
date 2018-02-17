from model.model import *
from view.view import View
from controller.controller import Controller
from model.parameters import *
import sys
import os
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


def createBots(number, model1):
    for i in range(number):
        model1.createBot()

if __name__ == '__main__':
    # This is used in case we want to use a freezing program to create an .exe
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    viewEnabled = int(input("Display view?: (1 == yes)\n"))
    viewEnabled = (viewEnabled == 1)

    model = Model(viewEnabled)

    numberOfBots = int(input("Please enter the number of bots:\n"))
    if numberOfBots == 0 and not viewEnabled:
        modelMustHavePlayers()
    if fitsLimitations(numberOfBots, MAXBOTS):
        createBots(numberOfBots, model)

    numberOfHumans = 0
    if viewEnabled:
        numberOfHumans = int(input("Please enter the number of human players: (" + str(MAXHUMANPLAYERS) + " max)\n"))
        if fitsLimitations(numberOfHumans, MAXHUMANPLAYERS):
            createHumans(numberOfHumans, model)
        if numberOfBots + numberOfHumans == 0:
            modelMustHavePlayers()

        if not model.hasHuman():
            spectate = int(input("Do want to spectate an individual bot's FoV? (1 = yes)\n"))
            if spectate == 1:
                model.addPlayerSpectator()

    screenWidth, screenHeight = defineScreenSize(numberOfHumans)
    model.setScreenSize(screenWidth, screenHeight)
    view = View(model, screenWidth, screenHeight)
    model.initialize()
    controller = Controller(model, viewEnabled, view)

    view.draw()

    while controller.running:
        controller.process_input()
        model.update()
