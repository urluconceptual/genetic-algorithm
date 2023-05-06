from math import log2, ceil
from numpy import random

# global variables
data = {}
population = {}
selections = {}
crossover = {}
mutation = []
currentStep = 0

fout = open('output.txt', 'w')
fout.write("output.txt\n")
fout.write("\n")


def readInput():
    global data
    fin = open('input.txt', 'r')

    # population size:
    n = int(fin.readline())
    # interval endpoints:
    left, right = list(map(int, fin.readline().split()))
    # coefficients of function:
    a, b, c = list(map(int, fin.readline().split()))
    # precision:
    p = int(fin.readline())
    # crossover probability:
    cp = int(fin.readline())/100
    # mutation probability:
    mp = int(fin.readline())/100
    # algorithm steps:
    s = int(fin.readline())

    fin.close()

    data = {"Population size": n, "Interval endpoints": [left, right], "Coefficients of function": [a, b, c],
            "Precision": p, "Crossover probability": cp, "Mutation probability": mp, "Number of steps": s}


def printInput():
    global data
    for k, v in data.items():
        v = v.__str__()
        fout.write(f"{f'{k}:':<30}{v:>30}\n")
    fout.write("\n")


def encode(n, a, d, l):
    # n is a string representing a real number
    n = float(n)
    # calculate which interval the number is part of
    x = (n - a)//d
    # return interval as binary
    return format(bin(int(x)).lstrip("0b"), f">0{l}")


def decode(n, a, d, p):
    # n is a string representing a real number in binary / which interval the number is part of
    x = int(n, 2)
    # return xth real number from interval:
    return round(a + x * d, p)


def generatePopulation():
    global data, population
    p = data["Precision"]
    e = data["Interval endpoints"]
    n = data["Population size"]
    c = data["Coefficients of function"]
    decimal = []
    binary = []
    f = []

    # calculate number of bits for representation
    l = ceil(log2((e[1] - e[0]) * (10**p)))
    # calculate distance between numbers
    d = (e[1] - e[0]) / (2**l)

    data["Number of bits"] = l
    data["Distance"] = d

    # generate random array of individuals
    randomPopulation = list(map(lambda x: round(x, p), random.uniform(e[0], e[1], n)))

    # encode the random population to fit the discrete intervals
    for i in range(n):
        bin = encode(randomPopulation[i], e[0], d, l)
        dec = decode(bin, e[0], d, p)
        fi = c[0] * (dec ** 2) + c[1] * dec + c[2]
        binary.append(bin)
        decimal.append(dec)
        f.append(fi)

        population = {"binary": binary, "decimal": decimal, "fitness": f}



def printPopulation():
    global population, data

    binary = population["binary"]
    decimal = population["decimal"]
    fitness = population["fitness"]
    n = data["Population size"]
    p = data["Precision"]

    fout.write("Population now:\n")

    for i in range(n):
        fout.write(f"{i + 1:>4}. b: {binary[i]}, d: {decimal[i]}, f: {fitness[i]}\n")

    fout.write("\n")


def generateProbabilityOfSelection():
    global population
    fitness = population["fitness"]
    F = sum(fitness)
    probabilityOfSelection = [f/F for f in fitness]
    population["probability of selection"] = probabilityOfSelection


def printProbabilityOfSelection():
    global population
    p = population["probability of selection"]

    fout.write("Probability of selection:\n")

    for i in range(len(p)):
        fout.write(f"chromosome {i + 1:>3} -> probability {p[i]}\n")

    fout.write("\n")


def generateIntervalsOfProbability():
    global population
    p = population["probability of selection"]

    intervals = [0]
    for i in range(1, len(p) + 1):
        intervals.append(intervals[i - 1] + p[i - 1])

    population["intervals of probability"] = intervals


def printIntervalsOfProbability():
    global population

    intervals = population["intervals of probability"]

    fout.write("Intervals of probability:\n")

    for i in intervals:
        fout.write(f"{i:>40}\n")

    fout.write("\n")


def generateSelection():
    global data, population, selections

    n = data["Population size"]
    intervals = population["intervals of probability"]
    uniforms = random.random_sample(n)
    selected = []

    for u in uniforms:
        l = 0
        r = len(intervals) - 1
        pos = 0

        while l <= r:
            m = (l + r) // 2
            if intervals[m] == u:
                pos = m - 1
                break
            elif intervals[m] < u:
                pos = m
                l = m + 1
            else:
                r = m - 1

        selected.append(pos + 1)

    selections["selections"] = uniforms
    selections["chromosomes"] = selected


def printSelections():
    global selections

    uniforms = selections["selections"]
    selected = selections["chromosomes"]


    fout.write("Selected chromosomes:\n")

    for i in range(len(uniforms)):
        fout.write(f"u = {uniforms[i]} => selection of chromosome {selected[i]}\n")

    fout.write("\n")


def select():
    global population, selections
    selected = selections["chromosomes"]
    binary = population["binary"]
    decimal = population["decimal"]
    fitness = population["fitness"]
    newBinary = []
    newDecimal = []
    newFitness = []
    for i in selected:
        newBinary.append(binary[i - 1])
        newDecimal.append(decimal[i - 1])
        newFitness.append(fitness[i - 1])

    population = {"binary": binary, "decimal": decimal, "fitness": fitness}


def generateProbabilityOfCrossover():
    global data, crossover
    n = data["Population size"]
    probability = data["Crossover probability"]
    uniforms = random.random_sample(n)

    isChosen = []
    chosen = []
    for i in range(len(uniforms)):
        if(uniforms[i] < probability):
            isChosen.append(True)
            chosen.append(i+1)
        else:
            isChosen.append(False)

    crossover = {"probabilities": uniforms, "isChosen": isChosen, "chosen": chosen}


def printCrossoverData():
    global crossover, population
    binary = population["binary"]

    fout.write("Selected for crossover:\n")

    for i in range(len(binary)):
        fout.write(f"chromosome {binary[i]:>3}, u = {crossover['probabilities'][i]} => {crossover['isChosen'][i]}\n")

    fout.write("\n")


def crossoverOperator(c1, c2):
    global data
    l = data["Number of bits"]
    point = random.randint(0, l-1)
    c1, c2 = [c1[:point] + c2[point:], c2[:point] + c1[point:]]
    return [point, c1, c2]


def doCrossover():
    global crossover, population, data
    chosen = crossover["chosen"]
    binary = population["binary"]
    decimal = population["decimal"]
    fitness = population["fitness"]
    a = data["Interval endpoints"][0]
    c = data["Coefficients of function"]
    d = data["Distance"]
    l = data["Number of bits"]
    p = data["Precision"]

    pairs = []
    crossovers = []
    while len(chosen) > 1:
        rand1 = 0
        rand2 = 0
        while rand1 == rand2:
            rand1 = random.choice(chosen)
            rand2 = random.choice(chosen)
        pairs.append((rand1, rand2))
        i1 = rand1 - 1
        i2 = rand2 - 1
        point, c1 , c2 = crossoverOperator(binary[i1], binary[i2])
        crossovers.append((binary[i1], binary[i2], point, c1, c2))
        decimal[i1] = decode(c1, a, d, p)
        decimal[i2] = decode(c2, a, d, p)
        binary[i1] = encode(decimal[i1], a, d, l)
        binary[i2] = encode(decimal[i2], a, d, l)
        fitness[i1] = c[0] * (decimal[i1] ** 2) + c[1] * decimal[i1] + c[2]
        fitness[i2] = c[0] * (decimal[i2] ** 2) + c[1] * decimal[i2] + c[2]
        chosen.remove(rand1)
        chosen.remove(rand2)

    crossover["pairs"] = pairs
    crossover["crossovers"] = crossovers
    population["binary"] = binary
    population["decimal"] = decimal
    population["fitness"] = fitness


def printCrossoverResults():
    global crossover
    pairs = crossover["pairs"]
    crossovers = crossover["crossovers"]

    for i in range(len(pairs)):
        fout.write(f"Crossover between chromosome {pairs[i][0]} and chromosome {pairs[i][1]}:\n")
        fout.write(f"             c1: {crossovers[i][0]}, c2:{crossovers[i][1]}, point: {crossovers[i][2]}\n")
        fout.write(f"    Result-> c1: {crossovers[i][3]}, c2:{crossovers[i][4]}\n")

    fout.write('\n')


def mutate(binary):
    i = random.randint(0, len(binary) - 1)
    binary = binary[:i] + ('1' if binary[i] == '0' else '0') + binary[i+1:]
    return binary


def doMutation():
    global population, mutation
    binary = population["binary"]
    decimal = population["decimal"]
    fitness = population["fitness"]
    a = data["Interval endpoints"][0]
    c = data["Coefficients of function"]
    d = data["Distance"]
    l = data["Number of bits"]
    p = data["Precision"]
    n = data["Population size"]
    probability = data["Mutation probability"]

    for i in range(n):
        if random.random_sample() < probability:
            binary[i] = mutate(binary[i])
            decimal[i] = decode(binary[i], a, d, p)
            fitness[i] = c[0] * (decimal[i] ** 2) + c[1] * decimal[i] + c[2]
            mutation.append(i+1)

    population["binary"] = binary
    population["decimal"] = decimal
    population["fitness"] = fitness


def printMutationResults():
    global mutation
    fout.write("Mutated chromosomes:\n")

    [fout.write(f"    chromosome {i}\n") for i in mutation]

    fout.write("\n")


def printResults():
    global population, currentStep
    decimal = population["decimal"]
    fitness = population["fitness"]
    maximum = max(fitness)
    mean = sum(fitness) / len(fitness)
    index = fitness.index(maximum)
    fout.write(f"Step {currentStep} -> MaxFitness: {maximum} at x = {decimal[index]}, MeanFitness: {mean}\n")


def intro():
    global currentStep
    currentStep += 1
    readInput()
    printInput()
    generatePopulation()
    printPopulation()
    generateProbabilityOfSelection()
    printProbabilityOfSelection()
    generateIntervalsOfProbability()
    printIntervalsOfProbability()
    generateSelection()
    printSelections()
    select()
    printPopulation()
    generateProbabilityOfCrossover()
    printCrossoverData()
    doCrossover()
    printCrossoverResults()
    printPopulation()
    doMutation()
    printMutationResults()
    printPopulation()
    printResults()


def evolution():
    global currentStep
    currentStep += 1
    generateProbabilityOfSelection()
    generateIntervalsOfProbability()
    generateSelection()
    select()
    generateProbabilityOfCrossover()
    doCrossover()
    doMutation()
    printResults()


def main():
    global currentStep
    intro()
    while currentStep < data["Number of steps"]:
        evolution()


main()


fout.close()
