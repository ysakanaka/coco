from __future__ import absolute_import, division, print_function
import numpy as np
import sys
from py4j.java_gateway import JavaGateway, CallbackServerParameters
from py4j.java_collections import SetConverter, MapConverter, ListConverter
sys.path.append('/Users/ysakanaka/Program/Research/Opt-IA_python')
import optIA


def opt_IA(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between
    `lbounds` and `ubounds`
    """
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None

    opt_ia = optIA.OptIA(fun, lbounds, ubounds)
    max_chunk_size = 1 + 4e4 / dim
    x_min = opt_ia.opt_ia(budget)

    return x_min

def opt_IA_reset_age(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between
    `lbounds` and `ubounds`
    """
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None

    opt_ia = optIA.OptIA(fun, lbounds, ubounds, ra=True)
    max_chunk_size = 1 + 4e4 / dim
    x_min = opt_ia.opt_ia(budget)

    return x_min

def opt_IA_search_assist(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between
    `lbounds` and `ubounds`
    """
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None

    opt_ia = optIA.OptIA(fun, lbounds, ubounds, ssa=True)
    max_chunk_size = 1 + 4e4 / dim
    x_min = opt_ia.opt_ia(budget)

    return x_min

def opt_IA_surrogate_assist(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between
    `lbounds` and `ubounds`
    """
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None

    opt_ia = optIA.OptIA(fun, lbounds, ubounds, sua = True)
    max_chunk_size = 1 + 4e4 / dim
    x_min = opt_ia.opt_ia(budget)

    return x_min


# ===============================================
# the most basic example solver
# ===============================================
def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between
    `lbounds` and `ubounds`
    """
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None
    max_chunk_size = 1 + 4e4 / dim
    while budget > 0:
        chunk = int(max([1, min([budget, max_chunk_size])]))
        # about five times faster than "for k in range(budget):..."
        X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim) # X is candidate array
        if fun.number_of_constraints > 0:
            C = [fun.constraint(x) for x in X]  # call constraints eval
            F = [fun(x) for i, x in enumerate(X) if np.all(C[i] <= 0)] # eval
        else:
            F = [fun(x) for x in X] # eval
        if fun.number_of_objectives == 1:
            index = np.argmin(F) if len(F) else None
            if index is not None and (f_min is None or F[index] < f_min):
                x_min, f_min = X[index], F[index]
        budget -= chunk
    return x_min
