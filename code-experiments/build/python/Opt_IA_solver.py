from __future__ import absolute_import, division, print_function
import numpy as np
from py4j.java_gateway import JavaGateway, CallbackServerParameters
from py4j.java_collections import SetConverter, MapConverter, ListConverter

class PythonEval:
    eval_fun_name = None
    eval_constraint_name = None
    x_min = 9999999
    budget = None
    chunk = None
    fun = None
    dim = 0

    def __init__(self, gateway):
        self.gateway = gateway

    def eval(self, obj):
         #print("obj", obj)
         x = None
         #x = np.empty(PythonEval.dim) ## TODO: check dimensions
         #np.delete(x, 0)
         for v in obj:
             #print("v is", v)
             x = np.append(x, v)

         x = np.delete(x, 0)
         #print("x is", x)
         f = None
         if PythonEval.fun.number_of_constraints > 0:
             c = PythonEval.fun.constraint(x)# call constraints eval
             if c <= 0 :
                 f = PythonEval.fun(x)
         else:
             f = PythonEval.fun(x)# eval
         #print("f is", f)
         return f

    def eval_all(self, obj):
        x = None
        xs = None
        print(obj)
        #java_list = ListConverter().convert(obj, self.gateway._gateway_client)
        for candidates in obj:
            for val in candidates:
                x = np.append(x, val)
                x = np.delete(x, 0)
            xs = np.append(xs, x)
        #print("xs", xs)
        xs = np.delete(xs, 0)
        #print("xs2", xs)

        f = None
        fs = []
        xval = None
        for c in obj:
            print("x",c)
            for v in c:
                #print("v is", v)
                xval = np.append(xval, v)
            xval = np.delete(xval, 0)
            print("xval", xval)
            if PythonEval.fun.number_of_constraints > 0:
                c = PythonEval.fun.constraint(xval)# call constraints eval
                if c <= 0 :
                    fs.append(PythonEval.fun(xval)) #???
            else:
                fs.append(PythonEval.fun(xval))# eval

            xval = None

        print(fs)
        java_list = ListConverter().convert(fs, self.gateway._gateway_client)
        return java_list

    def eval_constraint(self, obj):
        eval(PythonEval.eval_constraint_name)(self, obj)

    def set_eval(self, fun_name):
        PythonEval.eval_fun_name = fun_name

    def set_eval_constraint(self, fun_name):
        PythonEval.eval_constraint_name = fun_name

    def set_fun(self, fun):
        PythonEval.fun = fun

    def set_dim(self, dim):
        PythonEval.dim = dim

    def set_budget(self, b):
        PythonEval.budget = b

    def set_chunk(self, c):
        PythonEval.chunk = c

    def send_min(self, obj):
        #print("set min", obj)
        PythonEval.x_min = obj
        PythonEval.budget -= PythonEval.chunk
        #print("Budget",PythonEval.budget,PythonEval.chunk)
        if PythonEval.budget > 0:
            return 1
        else:
            return 0

    def fin(self):
        pass

    class Java:
        implements = ["main.PythonEvalFunc"]

def opt_IA(fun, lbounds, ubounds, budget, gateway):

    #gateway = JavaGateway(callback_server_parameters=CallbackServerParameters())
    pythonEval = PythonEval(gateway)
    #pythonEval.set_eval(fun.__name__)
    #pythonEval.set_eval_constraint(fun.constraint.__name__)
    pythonEval.set_fun(fun)

    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None
    #max_chunk_size = 1 + 4e4 / dim
    max_chunk_size = 20
    chunk = int(max([1, min([budget, max_chunk_size])]))

    pythonEval.set_dim(dim)
    #print("Future budget",budget)
    #print("lbounds", lbounds)
    #print("ubounds", ubounds)
    pythonEval.set_budget(budget)
    pythonEval.set_chunk(chunk)

    java_lbounds = gateway.jvm.java.util.ArrayList()
    java_ubounds = gateway.jvm.java.util.ArrayList()

    for l in lbounds:
        java_lbounds.append(l)

    for u in ubounds:
        java_ubounds.append(u)

    gateway.entry_point.init(pythonEval)
    gateway.entry_point.start(dim, java_lbounds, java_ubounds)

    #while pythonEval.budget > 0:
        #print("budget is " + pythonEval.budget)
    #gateway.shutdown()
    #print(pythonEval.x_min)
    return pythonEval.x_min



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
