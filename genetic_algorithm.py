import math
from random import choices, randint, randrange, random
from typing import Callable, List, Tuple
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
import optimization_functions as optF

Genome = List[float]
Population = List[Genome]
OptFunction = Callable[[Genome], float]

def generateGenome(length : int, a : float, b: float) -> Genome:
    return [np.random.uniform(a, b) for _ in range(length)]

def generatePopulation(size : int, genomeLength : int, a: float, b: float) -> Population:
    return [generateGenome(genomeLength, a, b) for _ in range(size)]

def simpleMutation(genome: Genome, a : float, b: float) -> Genome:
    i = randrange(len(genome))
    genome[i] = np.random.uniform(a, b)
    return genome

def biasedMutation(genome: Genome, a : float, b: float) -> Genome:
    return

def singlePointCrossover(p1: Genome, p2: Genome) -> Tuple[Genome, Genome]:
    i = randrange(1, len(p1)-1)
    return p1[0:i] + p2[i:], p1[0:i] + p2[i:]


def parentSelection(population: Population, optFunc : OptFunction, n: int) -> Population:
    return choices(
        population = population,
        weights = [optFunc(genome) for genome in population],
        k = n
    )

def sortPopulation(population: Population, fitnessFunc: OptFunction) -> Population:
    return sorted(population, key=fitnessFunc, reverse = False)

def geneticAlgorithm(
        popSize : int, 
        genomeLen : int,
        nGenerations : int,
        a : float,
        b : float,
        fitnessFunc : OptFunction) \
        -> Population:
    population = generatePopulation(popSize, genomeLen, a, b)

    #population += [[0,0,0,0,0]]
    for i in range(nGenerations):
        print(i)
        population = sortPopulation(population, fitnessFunc)

        if fitnessFunc(population[0]) == 0:
            break
        
        newGeneration = population[0:2]
        for j in range(int(len(population) / 2) - 1):
            parents = parentSelection(population, fitnessFunc, 2)
            offspringA, offspringB = singlePointCrossover(parents[0], parents[1])
            offspringA = simpleMutation(offspringA, a, b)
            offspringB = simpleMutation(offspringB, a, b)
            newGeneration += [offspringA, offspringB]

        population = newGeneration

    population = sortPopulation(population, fitnessFunc)
    print("\nAlgoritmo Genetico")
    print("- Tamaño de Genotipo:\t\t", genomeLen)
    print("- Tamaño de Población:\t\t", popSize)
    print("- Máximo de Generaciones:\t", nGenerations)

    print("- Función de Crossover:\t\t Single Point Crossover")
    print("- Función de Mutación:\t\t Simple Mutation")

    print("\n================Resultados================")
    if fitnessFunc(population[0]) == 0:
        print("Se encontró un óptimo en la generación", i+1)
    else:
        print("No se encontró un óptimo")

    print("\nMejor Resultado")
    print("Genotipo:", population[0])
    print("Fitness:", fitnessFunc(population[0]), "\n")

    return population

""" # Real Number range: a - lower bound, b - upper bound
a = 0
b = 10

newPop = generatePopulation(5, 5, a, b)
print(newPop)
print("Testing Mutation:\nPrevious Genome:", newPop[0])
simpleMutation(newPop[0], a, b)
print("New Genome:", newPop[0])
print("Testing Crossover:\nParents:", newPop[0], newPop[1], "\nOffSpring:", singlePointCrossover(newPop[0], newPop[1]))

newGen = [1,1,1,1,1,1,1,1]
print(rosenbrocksBanana(newGen))
 """
newPop = geneticAlgorithm(30, 5, 500000, -5.12, 5.12, optF.rastrigin)
print("\nResultados")
for i in range(0, len(newPop)-1):
    print(newPop[i], ":", optF.rastrigin(newPop[i]))
