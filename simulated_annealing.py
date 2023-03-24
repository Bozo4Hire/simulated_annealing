import math
import random
from random import choices, randint, randrange, random
from typing import Callable, List
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


Genome = List[int]
Population = List[Genome]

FitnessFunc = Callable[[Genome], int]

# implementacion de algoritmo de Fisher-Yates para generar lista aleatoria
def fisher_yates_shuffle(list : List[int]) -> List[int]:
    auxList = range(0, len(list))
    for i in list:
        j = randint(auxList[0], auxList[-1])
        list[i], list[j] = list[j], list[i]
    return list 

# generar un genoma de longitud n, donde cada elemento corresponde
# a la fila que ocupa la reina de cada columna. Utilizamos el algoritmo
# de Yates-fisher para acomodar el genotipo de forma aleatoria de tal 
# manera que ninguna reina comparta la misma fila
def generateGenome(length : int) -> Genome:
    return fisher_yates_shuffle( list(range(0, length)))

# generar a una poblacion de  (size) n individuos
def generatePopulation(size : int, genomeLength : int) -> Population:
    return [generateGenome(genomeLength) for _ in range(size)]

# funcion para evaluar fitness de un genoma
# 1 es el valor optimo de fitness, y 0 el peor valor posible
def fitness_function(genome : Genome) -> float :
    ft = len(genome) - 1

    #print("\nGenoma a evaluar:", genome)
    #print(genomeToStr(genome))

    for i in range(len(genome)):
        found_a = False
        found_b = False
        c = 1 
        for j in range(i+1, len(genome)):
            if genome[i] == genome[j]+c and found_a == False:
                ft -= 1
                found_a = True
            if genome[i] == genome[j]-c and found_b == False:
                ft -= 1 
                found_b = True
            c += 1
    #print ("valor de fitness: ", ft)
    return ft

def swappingMutation(genome: Genome) -> Genome:
    i = randrange(len(genome))
    j = randrange(len(genome))
    genome[i], genome [j] = genome[j], genome[i]
    return genome


def simulated_annealing(cost_function, initial_solution, temperature, cooling_rate, stopping_temperature):
    current_solution = initial_solution
    best_solution = initial_solution
    
    while temperature > stopping_temperature:
        # Generate a new solution by making a small change to the current solution
        new_solution = swappingMutation(current_solution)
        
        # Calculate the costs of the current and new solutions
        current_cost = cost_function(current_solution)
        new_cost = cost_function(new_solution)
        
        # Decide whether to accept the new solution or stay with the current solution
        if new_cost < current_cost:
            current_solution = new_solution
            if new_cost < cost_function(best_solution):
                best_solution = new_solution
        else:
            delta = new_cost - current_cost
            acceptance_probability = math.exp(-delta / temperature)
            if random() < acceptance_probability:
                current_solution = new_solution
        
        # Reduce the temperature according to the cooling rate
        temperature *= cooling_rate
        
    return best_solution

n = 8
initial_solution =  generateGenome(n)
print("Initial Solution: ", initial_solution)
print("Number of attacks:", fitness_function(initial_solution), "\n\n")
temperature = 100
cooling_rate = 0.95
stopping_temperature = 1e-8

best_solution = simulated_annealing(fitness_function, initial_solution, temperature, cooling_rate, stopping_temperature)

print("Best solution:", best_solution)
print("Number of attacks:", fitness_function(best_solution))
