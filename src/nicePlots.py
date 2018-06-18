from jobEvaluater import getAllDirectories, getDistributions
from modelCombiner import getMeanAndStDev, getTimeAxis
import matplotlib.pyplot as plt
import numpy

def getDesiredDirectories(desiredDirectories):
    return [directory for directory in getAllDirectories() if directory in desiredDirectories]


def produceAverageRun(inits):
    allMeanRuns = []
    for init in inits:
        runLength = len(init[0])
        meanRun = []
        for step in range(runLength):
            currentStep = []
            for run in init:
                currentStep.append(run[step])
            meanRun.append(numpy.mean(currentStep))
        allMeanRuns.append(meanRun)
    return allMeanRuns


def plot(name, dict):
    print("Plotting ", name, "...")


    colors = "rbcmgky"
    markers = "sovx8^D"
    if len(dict) > 6:
        print("Plotting is not (yet) supported for more than 6 curves! Add more markers and colors.")
        quit()


    fig, ax = plt.gcf(), plt.gca()
    #plt.clf()
    #print("dict is : ", dict)
    for idx, key in enumerate(dict):
        print("key: ", key)


        #print("value: ", dict[key])
        lineName = key
        dataList = dict[key]
        if name != "test" and name != "Pellet_Collection" and name != "VS_1_Greedy":
            dataList = produceAverageRun(dataList)
        dataLen = len(dataList[0])



        y, ysigma = getMeanAndStDev(dataList, dataLen)
        #print("Plotting: ", y)


        lenY = len(y)

        maxPoints = 150
        if lenY > maxPoints:
            meanStep = int(lenY / maxPoints)
            y = [numpy.mean(y[idx:idx+meanStep]) for idx in range(0, lenY, meanStep)]
            ysigma = numpy.array([numpy.mean(ysigma[idx:idx+meanStep]) for idx in range(0, lenY, meanStep)])
            lenY = len(y)
        else:
            meanStep = 1

        y_lower_bound = y - ysigma
        y_upper_bound = y + ysigma

        color = colors[idx]
        marker = markers[idx]
        plt.ticklabel_format(axis='x', style='sci', scilimits=(1, 4))
        maxMarkers = 10
        if name == "test" or name == "Pellet_Collection" or name == "VS_1_Greedy":
            x = range(0, 105, 5)
        else:
            x = range(0, lenY * meanStep, meanStep)

        if lenY > maxMarkers:

            smallPartIdxs = range(0,(lenY * meanStep),int(lenY / maxMarkers * meanStep))
            if name == "test" or name == "Pellet_Collection" or name == "VS_1_Greedy":
                smallPartIdxs = range(0, 101, int(100/maxMarkers))

            smallPart = y[0:(lenY):int(lenY / maxMarkers)]
            plt.plot(smallPartIdxs, smallPart, label=lineName, color=color, marker=marker, markersize=10,
                     linestyle='None', markerfacecolor=color, markeredgecolor=color)
            plt.plot(x, y, lw=2, color=color)
        else:
            plt.plot(x, y, lw=2, label=lineName, color=color, marker=marker, markersize=10)
        ax.fill_between(x, y_lower_bound, y_upper_bound, facecolor=color, alpha=0.35)

    if name == "test" or name == "Pellet_Collection" or name == "VS_1_Greedy":
        ax.set_xlabel("Percentage of training time")
    else:
        ax.set_xlabel("Testing Steps")
    ax.set_ylabel("Mass")
    plt.title(name)

    ax.legend(loc='upper left')

    ax.grid()
    fig.savefig("savedModels/" + name +  ".pdf")
    plt.close()


def getDictDict(distributions):
    dictDict = {} #contains a list of dictionaries that need plotting
    for dictionary in distributions:
        name = dictionary["name"] # name of the directory
        for key in dictionary:
            if key == "name":
                continue

            data = dictionary[key]
            try:
                dictDict[key]
            except KeyError:
                dictDict[key] = {}
            dictDict[key][name] = data
    return dictDict




def plotDict(dictDict):
    for key in dictDict:
        plot(key, dictDict[key])

if __name__ == '__main__':
    directoriesToPlot = []
    print("Available directories:")
    for directory in getAllDirectories():
        print(directory)
    dirName = input("Type the name of the directory you want to include in the plot: ")
    directoriesToPlot.append(dirName)
    while input("More? (y==more)\n") == "y":
        directoriesToPlot.append(input("Type in the name: "))

    dirs = getDesiredDirectories(directoriesToPlot)

    #print("Dirs: ", dirs)

    distributions = getDistributions(dirs, takeMean=False)

    #print("Distributions: ", distributions)

    dictDict = getDictDict(distributions)

    #print("DictDict: ", dictDict)

    plotDict(dictDict)



























