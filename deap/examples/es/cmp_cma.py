# -*- coding: utf-8 -*-

#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import numpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

# Problem size
N =40
NGEN = 330000/800
#NGEN = 100000/400
#NGEN = 150000/600

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#toolbox.register("evaluate", benchmarks.rastrigin)
toolbox.register("evaluate", benchmarks.rastrigin)
def gendata(para, verbose=True):
    global NGEN
    # The cma module uses the numpy random number generator
    numpy.random.seed(128)

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES    
    strategy = cma.Strategy(centroid=[5.0]*N, sigma=5.0, lambda_=20*N)
    if para == 1:
      
        strategy.setParams(0.18596345254731422, 0.039962574520232684, 0.5238280816214763)
       

    
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    # Objects that will compile the data
    sigma = numpy.ndarray((NGEN,1))
    axis_ratio = numpy.ndarray((NGEN,1))
    diagD = numpy.ndarray((NGEN,N))
    fbest = numpy.ndarray((NGEN,1))
    best = numpy.ndarray((NGEN,N))
    std = numpy.ndarray((NGEN,N))

    for gen in range(NGEN):
        print(gen)
        print(para)
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Update the strategy with the evaluated individuals
        toolbox.update(population)
        
        # Update the hall of fame and the statistics with the
        # currently evaluated population
        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(evals=len(population), gen=gen, **record)
        
        if verbose:
            print(logbook.stream)
        
        # Save more data along the evolution for latter plotting
        # diagD is sorted and sqrooted in the update method
        sigma[gen] = strategy.sigma
        axis_ratio[gen] = max(strategy.diagD)**2/min(strategy.diagD)**2
        diagD[gen, :N] = strategy.diagD**2
        fbest[gen] = halloffame[0].fitness.values
        best[gen, :N] = halloffame[0]
        std[gen, :N] = numpy.std(population, axis=0)

    # The x-axis will be the number of evaluations
    
    avg, max_, min_ = logbook.select("avg", "max", "min")
    return avg, max_, min_, fbest
def plot1(y1,NGEN):
    x = list(range(0,20*N*NGEN,20*N))
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.semilogy(x, y1, "-b")
    #plt.semilogy(x, y2, "-r")
    plt.grid(True)
    plt.title("f: sphere  blue: cma  red: tpe_cma")
    plt.show()
def plot2(y2,NGEN):
    x = list(range(0, NGEN))
    plt.figure()
    plt.subplot(1, 1, 1)
    #plt.semilogy(x, y1, "-b")
    plt.semilogy(x, y2, "-r")
    plt.grid(True)
    plt.title("f: sphere  blue: cma  red: tpe_cma")
    plt.show()
def plot(y1,y2):
    x = list(range(0,20*N*NGEN,20*N))
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.semilogy(x, y1, "-b")
    plt.semilogy(x, y2, "-r")
    
    plt.grid(True)
    plt.autoscale()
    plt.xlabel("Number of function evaluations")
    plt.ylabel("log10(Objective function")
    #plt.title("Sphere 40-D")rastrigin
    #plt.title("Sphere 20-D")
    plt.title("Rastrigin 40-D")
    plt.show()
    
    
if __name__ == "__main__":
    
    
    #avg, mmax_, mmin_, fbest2 = gendata(2,False)
    avg, mmax_, mmin_, r1 = gendata(0,False)
    avg, max_, min_, r2= gendata(1,False)
    s1 = []
    s2 = []
    s1.append(r1[-1,0])
    s2.append(r2[-1,0])
    for i in xrange(14):
        avg, mmax_, mmin_, fbest0= gendata(0,False)
        r1 = r1 + fbest0
        s1.append(fbest0[-1,0])
        avg, max_, min_, fbest1= gendata(1,False) 
        r2 = r2 + fbest1
        s2.append(fbest1[-1,0])
    #avg, mmax_, mmin_, fbest4 = gendata(4,False)
    plot(r1/15.0, r2/15.0)
    print(numpy.mean(s1))
    print(numpy.std(s1))
    print(numpy.mean(s2))
    print(numpy.std(s2))


