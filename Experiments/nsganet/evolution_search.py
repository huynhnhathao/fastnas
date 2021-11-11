from functools import lru_cache
import sys
sys.path.insert(0,r'C:\Users\ASUS\Desktop\fast-nas' )
import os
import time
import logging
import argparse

import numpy as np

from pymop.problem import Problem
from pymoo.optimize import minimize
from nats_bench import create

import nsganet
import utils
from genotypes import Structure

# argparse 
parser = argparse.ArgumentParser("NAS with NSGA II and NATS_bench")

parser.add_argument('--pop_size', type = int, default = 40, help = 'population size for the NSGA algorithm')
parser.add_argument('--n_gens', type = int, default= 50, help = 'number of generations to run')
parser.add_argument('--n_offspring', type = int, default = 40, help = 'number of offspring created each generation')
parser.add_argument('--save', type = str, default='experiment_results',help = 'experiment name' )
parser.add_argument('--seed', type = int, default = 42, help = 'random seed')


args = parser.parse_args()
# args.save = 'search-{}-{}-{}-{}'.format(args.save, args.seed, time.strftime('%Y%m%d-%H%M%S'))
# utils.create_exp_dir(args.save)

#logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream = sys.stdout, level = logging.INFO,
                     format = log_format, datefmt= '%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
logger = logging.getLogger()#.addHandler(fh)

operations = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
api = create(r'C:\Users\ASUS\Desktop\nas\nas-without-training\NATS-tss-v1_0-3ffb9-simple', 'tss', True, verbose= False)

def genome_to_genotypes(genome:list) -> list:
    """
    decode genome to genotypes
    """
    # there are 6 edges in each cell
    assert len(genome) == 6
    genotypes = [((operations[genome[0]], 0),  ), ( (operations[genome[1]], 0), (operations[genome[2]], 1) ), 
                ( (operations[genome[3]], 0), (operations[genome[4]], 1), (operations[genome[5]], 2))]
    return genotypes


def query_performance(genome:list, metrics:list = ['FLOPs', 'val-acc'], dataset = 'cifar100') -> tuple:
    """
    query the performance of the architecture constructed by the genome
    """
    genotypes = genome_to_genotypes(genome)
    network_string = Structure(genotypes).tostr()
    print(network_string)
    network_index = api.query_index_by_arch(network_string)
    print(network_index)
    cost_info = api.get_cost_info(network_index, dataset)
    
    more_info = api.get_more_info(network_index, dataset = dataset, hp = '200')

    performance = {'FLOPs': cost_info['flops'], 'valid-accuracy': more_info['valid-accuracy']}
    return performance


class NAS(Problem):
    def __init__(self, n_var, n_obj, n_constr = 0, lb = None, ub = None,save_dir = None):
        super().__init__(n_var = n_var, n_obj = n_obj, n_constr = n_constr, type_var = np.int)
        self.xl = lb
        self.xu = ub
        self.save_dir = save_dir
        self._n_evaluated = 0

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_objs), np.nan)

        for i in range(x.shape[0]):
            # Now how do you evaluate the architecture
            arch_id = self._n_evaluated + 1
            logger.info(f'Network id = {arch_id}')

            performance = query_performance(genome = x[i, :], metrics = ['FLOPs', 'valid-accuracy'])
            objs[i, 0] = performance['FLOPs']
            objs[i, 1] = 100 - performance['valid-accuracy']
            self._n_evaluated +=1

        out['F'] = objs

def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # report generation info to files
    logging.info("generation = {}".format(gen))
    logging.info("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    logging.info("population complexity: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                  np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))



if __name__ == '__main__':
    # prepare the nats api

    np.random.seed(args.seed)
    logging.info('args = %s', args)

    n_var = 12 
    lb = np.zeros(n_var)
    ub = np.ones(5)*4
    problem = NAS(n_var = n_var, n_obj = 2, n_constr = 0, lb = lb, ub = ub)
    algorithm = nsganet.nsganet(pop_size = args.pop_size, n_offsprings = args.n_offspring, 
                                eliminate_duplicates = True)

    res = minimize(problem, algorithm,
                    callback = do_every_generations,
                    termination= ('n_gens', args.n_gens))
    logger.info(res)