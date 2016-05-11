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






# Problem size
N=30
p_rastrigin = None
mmin = 10000
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#rosenbrock
toolbox.register("evaluate", benchmarks.rastrigin)


def cma_es(params):
    global mmin,p_rastrigin
    # The cma module uses the numpy random number generator
    numpy.random.seed(128)
    cc, ccov1, ccovmu = params
    
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
    algorithms.eaGenerateUpdate(toolbox, ngen=500, stats=stats, halloffame=hof)

    # print "Best individual is %s, %s" % (hof[0], hof[0].fitness.values)
    
    if hof[0].fitness.values[0] < mmin :
        mmin = hof[0].fitness.values[0]
        p_rastrigin = params
    return hof[0].fitness.values[0]

best = fmin(cma_es,
            space=[hp.uniform('cc', 0, 1),
                   hp.uniform('ccov1', 0, 1),
                   hp.uniform('ccovmu', 0, 1)],
            algo=tpe.suggest,
            max_evals=50)
print(best)
print(mmin)

print(p_rastrigin)


