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
from hyperopt import fmin, tpe, hp
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools
import math
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample
import random

#########################
a=0

def foo(x):
    print("a")
    print(a)
    b=random.uniform(0,1-a)
    print("b")
    print(b)
    return b
def foo1(x):
    
    global a
    a = x
    print("x")
    print(x)
    return x

################################ Problem size
count = 0
p = None
mmin = 10000000

def cma_es(params, N, bench, la):
    
   
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    #rosenbrock
    toolbox.register("evaluate", bench)
    # The cma module uses the numpy random number generator
    numpy.random.seed(128)
    cc, ccovmu , ccov1 = params
    
    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES    
    strategy = cma.Strategy(centroid=[5.0]*N, sigma=5.0, lambda_=20*N)
    strategy.setParams(cc, ccov1, ccovmu)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    #logger = tools.EvolutionLogger(stats.functions.keys())
   
    # The CMA-ES algorithm converge with good probability with those settings
    algorithms.eaGenerateUpdate(toolbox, ngen=50, stats=stats, halloffame=hof)

    # print "Best individual is %s, %s" % (hof[0], hof[0].fitness.values)
    return hof[0].fitness.values[0]

def train_cam(params):
    global p
    global mmin
    global count
    count += 1
    print(count)
    print(params)
    cc, ccov1, ccovmu = params
    if ccov1+ccovmu > 1:
        return 10000000
    sum_ = 0
    for i in range(6):
         flag = random.randint(1,12)
         print(flag)
         if flag ==1 or flag == 2 or flag==3:
             if flag == 3: flag += 1
             sum_ += cma_es(params,10*flag,benchmarks.sphere,flag)
         if flag ==4 or flag == 5 or flag == 6:
             flag -= 3
             if flag == 3: flag += 1
             sum_ += cma_es(params,10*flag,benchmarks.cigar,flag)
         if flag ==7 or flag == 8 or flag == 9:
             flag -= 6
             if flag == 3: flag += 1
             sum_ += cma_es(params,10*flag,benchmarks.rosenbrock,flag)
         if flag ==10 or flag == 11 or flag == 12:
             flag -= 9
             if flag == 3: flag += 1
             sum_ += cma_es(params,10*flag,benchmarks.rastrigin,flag)
    print(sum_)
    if  (sum_/6.0) <= mmin :
        mmin = sum_/6.0
        p = params
        print(p)
    return sum_/6.0
             

if __name__ == "__main__":
    
    expr_space = {
        'a': scope.call(foo, args=(random.uniform(0, 1),)),
        'b': scope.call(foo1, args=(random.uniform(0, 1),)),
        'c': hp.uniform('c', 0, 1),
    }
    
    best = fmin(train_cam,
                space=expr_space,
                algo=tpe.suggest,
                max_evals=1000)
    print(best)
    print(p)
    print(mmin)
