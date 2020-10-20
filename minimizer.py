'''Minimizer routine based on scipy.optimize.minimize'''

import pickle
import numpy as np
from scipy.optimize import minimize
from numpy import linalg as LA

class Minimizer():
    """
    A class to minimize an objective function
    give an f(0) and constraints
    """
    def __init__(self, n_sensors=127, alpha=0.01, S=None):
        
        self.n_sensors = n_sensors
        self.alpha = alpha
        self.S = S
        """
        scipy uses a list of objects specifying constraints
        to the optimization problem.
        Inequality means that it is to be non-negative
        """
        con1 = con1 = {'type': 'ineq', 'fun': self.constraint1}
        con2 = {'type': 'ineq', 'fun': self.constraint2}

        self.cons = ([con1, con2])

    def to_vector(self, L):
        """
        scipy.optimize.minmize uses 1D vectors,
        therefore we flat the matrix
        (this is just a workaround, please provide input if you can)
        param L: Laplacian
        return: flatten Laplacian
        """
        assert L.shape == (self.n_sensors, self.n_sensors) 
        return L.flatten()

    def to_matrix(self, vec):
        """
        scipy.optimize.minmize uses 1D vectors
        param vec: 1D vector
        return: matrix
        """
        assert vec.shape == (self.n_sensors*self.n_sensors, )
        return vec[:self.n_sensors*self.n_sensors].reshape(self.n_sensors, self.n_sensors)
    
    def objective_function(self, L):
        """
        objective function
        param L: Laplacian (see, https://arxiv.org/abs/1601.02513)
        return: objective_function
        """
        if L.shape != (self.n_sensors, self.n_sensors):
            L = self.to_matrix(L)

        tr = np.trace(np.matmul(self.S, L))
        return tr + self.alpha*LA.norm(L, 'fro')

    def constraint1(self, L):
        """
        constraint trace(L)>0, on https://arxiv.org/abs/1601.02513
        trace(L)>s where s is the number of nodes
        param L: Laplacian
        return: trace(L)
        """
        if L.shape != (self.n_sensors, self.n_sensors):
            L = self.to_matrix(L)

        tr = np.trace(L)
        return tr

    def constraint2(self, L):
        """
        constraint tr + alpha*LA.norm(L, 'fro')>0,
        objective function must be positive
        param L: Laplacian
        return: constraint function
        """
        if L.shape != (self.n_sensors, self.n_sensors):
            L = self.to_matrix(L)

        tr = np.trace(np.matmul(self.S, L))
        return tr + self.alpha*LA.norm(L, 'fro')

    def Optimization(self, L0, maxiter):
        """
        Optimization, method a Trust region
        param L0: initial guess
        param maxiter: maximum of iterations
        return: result (Laplacian)
        """        
        result = minimize(self.objective_function, self.to_vector(L0),
                          method='trust-constr',
                          constraints=self.cons,
                          options={'maxiter': maxiter, 'verbose': 3, 'gtol': 1e-10})
        
        result.x = self.to_matrix(result.x)
        return result
