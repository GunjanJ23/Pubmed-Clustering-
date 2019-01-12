import math
from math import *
import sys
from random import shuffle, uniform
import matplotlib.pyplot as plt


def ReadData(fileName):
    f = open(fileName, 'r')
    lines = f.read().splitlines()
    items = []

    for i in range(1, len(lines)):
        line = lines[i].split()
        itemFeatures = []
        v = float(line[0])
        itemFeatures.append(v)
        items.append(itemFeatures)

    shuffle(items)
    return items


def FindColMinMax(items):
    minima = [sys.maxsize for i in range(len(items[0]))]
    maxima = [-sys.maxsize-1 for i in range(len(items[0]))]

    for item in items:
        for f in range(len(item)):
            if (item[f] < minima[f]):
                minima[f] = item[f]

            if (item[f] > maxima[f]):
                maxima[f] = item[f]

    return minima, maxima


def EuclideanDistance(x, y):
    summ = 0
    for i in range(len(x)):
        summ+= math.pow(x[i] - y[i], 2)
    return math.sqrt(summ)


def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)

def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)

def InitializeMeans(items, k, cMin, cMax):
    f = len(items[0])
    means = [[0 for i in range(f)] for j in range(k)]

    for mean in means:
        for i in range(len(mean)):
            mean[i] = uniform(cMin[i] - 1, cMax[i] + 1)
    return means


def UpdateMean(n, mean, item):
    for i in range(len(mean)):
        m = mean[i]
        m = (m * (n - 1) + item[i]) / float(n)
        mean[i] = round(m, 3)

    return mean


def FindClusters(means, items):
    clusters = [[] for i in range(len(means))]

    for item in items:
        index = Classify(means, item)
        clusters[index].append(item)

    return clusters


def Classify(means, item):

    minimum = sys.maxsize
    index = -1

    for i in range(len(means)):
        dist = EuclideanDistance(item, means[i])
        # dist = manhattan_distance(item,means[i])
        # dist = jaccard_similarity(item,means[i])
        # dist = cosine_similarity(item,means[i])

        if (dist < minimum):
            minimum = dist
            index = i
    return index


def CalculateMeans(k, items, maxIterations=100):
    cMin, cMax = FindColMinMax(items)

    # Initialize means
    means = InitializeMeans(items, k, cMin, cMax)

    clusterSizes = [0 for i in range(len(means))]
    belongsTo = [0 for i in range(len(items))]


    for e in range(maxIterations):
        noChange = True
        for i in range(len(items)):
            item = items[i]
            index = Classify(means, item)

            clusterSizes[index] += 1
            means[index] = UpdateMean(clusterSizes[index], means[index], item)

            if (index != belongsTo[i]):
                noChange = False

            belongsTo[i] = index

        if (noChange):
            break

    return means


def normalize(input):
    summ = 0
    for index in range(len(input)):
        summ += input[index][0]

    for index in range(len(input)):
        input[index] = [(input[index][0] / summ) * 1000]
    return (input,summ)


def normalizedFile(file):
    input = []
    for line in file:
        for pos, word in enumerate(line.split()):
            if pos == 0:
                input.append([int(word)])
    return(normalize(input))

def label(file):
    labels = {}
    for line in file:
        for pos, word in enumerate(line.split()):
            if pos == 0:
                labels[int(word)] = ''
                id = int(word)
                continue
            labels[id] += word
    return labels

def testFile(file):
    valInput = []
    for word in file:
        valInput.append([int(word)])
    return(normalize(valInput))


def results(labels,op,valInput,valsumm):
    l = list(set(labels.values()))

    for key, value in labels.items():
        labels[key] = l.index(value)

    true_pred = {}
    for key, value in labels.items():
        key = (key / valsumm) * 1000
        if value in true_pred:
            true_pred[value].append(key)
        else:
            true_pred[value] = [key]

    opFile = {}
    for index in range(len(op)):
        if op[index] in opFile:
            opFile[op[index]].append(valInput[index][0])
        else:
            opFile[op[index]] = [valInput[index][0]]

    return (true_pred,opFile)


def truthTable(labels,valsumm):
    l = list(set(labels.values()))

    for key, value in labels.items():
        labels[key] = l.index(value)

    true_pred = {}
    for key, value in labels.items():
        key = (key / valsumm) * 1000
        if value in true_pred:
            true_pred[value].append(key)
        else:
            true_pred[value] = [key]

    return true_pred


def plot(true_pred,opFile):
    idOp = {}
    for key, value in opFile.items():
        for i in value:
            idOp[i] = key
    result = {}
    for key, value in true_pred.items():
        for i in value:
            if i in idOp:
                result[i] = (key, idOp[i])

    # plotting
    colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'c', 6: 'm'}
    for key, value in result.items():
        plt.scatter(key, value[0], c=colmap[value[0] + 1])
        plt.scatter(key, value[1], c=colmap[6 - value[1]], marker='^')
    plt.xlim([0, 12])
    plt.ylim([0, 6])
    plt.xlabel('Gold_Id')
    plt.ylabel('Category')
    plt.show()

    return(result)

def quant(true_pred,opFile):
    clusters = 6
    quant = {}
    for i in range(clusters):
        quant[i] = (len(true_pred[i]), len(opFile[i]))
    return(quant)


def main(file):
    noclusters = 6
    #training file
    trainFile = open(file, 'r')

    #input
    input, summ = normalizedFile(trainFile)
    means = CalculateMeans(noclusters, input)

    #trueLabels
    trainFile = open(file,'r')
    labels = label(trainFile)

    #validation file
    valFile = open('pmids_gold_set_unlabeled.txt', 'r')
    valInput, valsumm = testFile(valFile)

    # Training k means
    clusters = FindClusters(means, valInput)

    # Results
    opFile = {}
    for i in range(len(clusters)):
        opFile[i] = []
        for item in clusters[i]:
            opFile[i].append(item[0])

    true_pred = truthTable(labels, valsumm)

    #plotting

    result = plot(true_pred,opFile)

    #Quantitative result
    print(quant(true_pred,opFile))


    #Testing
    #
    test= open('pmids_test_set_unlabeled.txt', 'r')
    #Preprocessing
    testInput, testsumm = testFile(test)
    #output
    clusters = FindClusters(means, testInput)
    op = {}
    for i in range(len(clusters)):
        op[i] = []
        for item in clusters[i]:
            op[i].append(item[0])
    #
    idOp = {}
    for key, value in op.items():
        for i in value:
            idOp[i] = key


    #plotting
    colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'c', 6: 'm'}
    for index in range(len(testInput)):
        plt.scatter(testInput[index][0], idOp[testInput[index][0]], c=colmap[idOp[testInput[index][0]] + 1])
    plt.xlim([0, 15])
    plt.ylim([0, 6])
    plt.xlabel('Test_Gold_Id')
    plt.ylabel('Category')
    plt.show()

    test= open('pmids_test_set_unlabeled.txt', 'r')
    outputFile = {}
    for pos,line in enumerate(test):
        key = (int(line)/testsumm)*1000
        outputFile[int(line)] = idOp[key]



    f1 = open('pmids_test_set_labeled.txt', 'w+')
    for k, v in outputFile.items():
        f1.write(str(k) + '\t' +str(v) +'\n')
    f1.close()


if __name__ == "__main__":
    path = sys.argv[1]
    main(path)