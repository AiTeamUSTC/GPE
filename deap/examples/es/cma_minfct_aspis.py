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
import GPy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools
from apsis.models.parameter_definition import MinMaxNumericParamDef
from apsis.assistants.lab_assistant import LabAssistant
from apsis.optimizers.bayesian.acquisition_functions import ExpectedImprovement
# Problem size
N=30

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("evaluate", benchmarks.sphere)

def cma_es(cc, ccov1, ccovmu):
    # The cma module uses the numpy random number generator
    numpy.random.seed(128)
  
    
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
    algorithms.eaGenerateUpdate(toolbox, ngen=200, stats=stats, halloffame=hof)

    # print "Best individual is %s, %s" % (hof[0], hof[0].fitness.values)
    
    return hof[0].fitness.values[0]

if __name__ == "__main__":
    param_defs = {
        'cc': MinMaxNumericParamDef(0, 1),
        'ccov1': MinMaxNumericParamDef(0, 1),
        'ccovmu': MinMaxNumericParamDef(0, 1)
    }
    assistant = LabAssistant()
    print(assistant.init_experiment("bay_EI", "BayOpt", param_defs, exp_id="cmaes",  minimization=True, optimizer_arguments={"kernel": GPy.kern.Matern52, "kernel_params": {"ARD": True},"acquisition": ExpectedImprovement, "initial_random_runs": 5} ))

    #assistant.init_experiment("cmaes", "BayOpt", param_defs, minimization=True, exp_id="cmaes")
    for i in range(100):
        candidate = assistant.get_next_candidate("cmaes")
        if candidate == None:
            continue
        cc = candidate.params['cc']
        ccov1 = candidate.params['ccov1']
        ccovmu = candidate.params['ccovmu']
        print(cc)
        print(ccov1)
        print(ccovmu)
        candidate.result = cma_es(cc, ccov1, ccovmu)
        assistant.update("cmaes", "finished", candidate)
    best_cand = assistant.get_best_candidate("cmaes")
    print("Best result: " + str(best_cand.result))
    print("with parameters: " + str(best_cand.params))
