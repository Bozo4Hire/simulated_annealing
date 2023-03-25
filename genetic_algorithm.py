import math
from random import choices, randint, randrange, random
from typing import Callable, List, Tuple
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
from optimization_functions import rastrigin, threeHumpCamel, ackley, rosenbrocksBanana

Genome = List[float]
Population = List[Genome]

def generateGenome(length : int, a : float, b: float) -> Genome:
    return [np.random.uniform(a, b) for _ in range(length)]

def generatePopulation(size : int, genomeLength : int, a: float, b: float) -> Population:
    return [generateGenome(genomeLength, a, b) for _ in range(size)]

def simpleMutation(genome: Genome, a : float, b: float) -> Genome:
    i = randrange(len(genome))
    genome[i] = np.random.uniform(a, b)
    return genome

def singlePointCrossover(p1: Genome, p2: Genome) -> Tuple[Genome, Genome]:
    i = randrange(1, len(p1)-1)
    return p1[0:i] + p2[i:], p1[0:i] + p2[i:]

# Real Number range: a - lower bound, b - upper bound
a = 0
b = 10

newPop = generatePopulation(5, 5, a, b)
print(newPop)

print("Testing Mutation:\nPrevious Genome:", newPop[0])
simpleMutation(newPop[0], a, b)
print("New Genome:", newPop[0])
print("Testing Crossover:\nParents:", newPop[0], newPop[1], "\nOffSpring:", singlePointCrossover(newPop[0], newPop[1]))

newGen = [1,1,1,1,1,1,1,1]

#print(threeHumpCamel(newGen))
#print(ackley(newGen))
print(rosenbrocksBanana(newGen))
