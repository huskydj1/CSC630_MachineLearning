import numpy as np
import math
# TODO: TYPE CHECKING IN OPERATOR OVERLOAD
# TODO: Add docstrings
# TODO: Printing computational graph: https://stackoverflow.com/questions/22920433/python-draw-flowchart-illustration-graphs
class Variable():

    independentvariables = []

    @ staticmethod
    def onehotvector(name):
        index = Variable.indexinvector(name)

        gradient_vector = np.zeros(shape = len(Variable.independentvariables))
        
        gradient_vector[index] = 1

        return gradient_vector

    @ staticmethod
    def indexinvector(name):
        Variable.independentvariables.sort()
        assert len(set(Variable.independentvariables)) == len(Variable.independentvariables)

        for i, val in enumerate(Variable.independentvariables):
            if val==name:
                return i

        return -1

    def __init__(self, name = None, evaluate = None, grad = None):
        
        # Store Independent Variable Name 
        if name != None: # Name exists if and only if instance is an independent variable
            self.name = name
            Variable.independentvariables.append(name)
        
        # Setting node's operation
        if evaluate == None: # Independent Variable
            self.evaluate = lambda values : values[name]
            self.grad = lambda values : Variable.onehotvector(name)
            
        else: # Operator Node
            self.evaluate = evaluate 
            self.grad = grad
        
    
    def __call__(self, **kargs):
        #print(type(kargs))
        #print(kargs) 
        return self.evaluate(kargs)
    
    def gradient(self, **kargs):
        return self.grad(values = kargs)

    def __add__(self, other):
        if isinstance(other, Variable): # Adding with Variable Instance
            return Variable(
                name = None, 
                evaluate = lambda values : self.evaluate(values) + other.evaluate(values),
                grad = lambda values : self.grad(values) + other.grad(values),
            )
        else: # Adding with constant
            return Variable(
                name = None, 
                evaluate = lambda values : self.evaluate(values) + other,
                grad = lambda values : self.grad(values)
            )

    
    def __radd__(self, other):
        return self.__add__(other)            

    def __mul__(self, other):
        if isinstance(other, Variable): # Multiplying with Variable Instance
            return Variable(
                name = None, 
                evaluate = lambda values : self.evaluate(values) * other.evaluate(values),
                grad = lambda values : self.evaluate(values) * other.grad(values) + self.grad(values) * other.evaluate(values),
            )
        else: # Multiplying with constant
            return Variable(
                name = None, 
                evaluate = lambda values : self.evaluate(values * other),
                grad = lambda values : other * self.grad(values),
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        if isinstance(other, Variable): # Power is Variable Instance
            return Variable(
                name = None, 
                evaluate = lambda values : pow(self.evaluate(values), other.evaluate(values)),
                grad = lambda values : pow(self.evaluate(values), other.evaluate(values)) * (other.__mul__(Variable.ln(self))).grad 
           )
        else: # Power is a constant
            return Variable(
                name = None, 
                evaluate = lambda values : pow(self.evaluate(values), other),
                grad = lambda values : pow(self.evaluate(values), other) * (Variable.ln(self).mul(other)).grad 
            )

    def __rpow__(self, other):
        if isinstance(other, Variable): # Power is variable, base is constant
            return Variable(
                name = None,
                evaluate = lambda values : pow(other, self.evaluate(values)),
                grad = lambda values : pow(other, self.evaluate(values)) * (self.__mul__(math.ln(other))).grad
            )
        else: # Power is variable, base is variable
            return Variable(
                name = None, 
                evaluate = lambda values : pow(other.evaluate(values), self.evaluate(values)),
                grad = lambda values : pow(other.evaluate(values), self.evaluate(values)) * (self.__mul__(Variable.ln(other))).grad 
           )
    
    def __sub__(self, other):
        if isinstance(other, Variable): # Subtracting with Variable Instance
            return Variable(
                name = None, 
                evaluate = lambda values : self.evaluate(values) - other.evaluate(values),
                grad = lambda values : self.grad(values) - other.grad(values)
            )
        else: # Subtracting with Constant
            return Variable(
                name = None,
                evaluate = lambda values : self.evaluate(values) - other,
                grad = lambda values : self.grad(values)
            )

    def __rsub__ (self, other):
        if isinstance(other, Variable): # Subtracting with Variable Instance
            return Variable(
                name = None, 
                evaluate = lambda values : other.evaluate(values) - self.evaluate(values),
                grad = lambda values : other.grad(values) - self.grad(values)
            )
        else: # Subtracting with Constant
            return Variable(
                name = None,
                evaluate = lambda values : other - self.evaluate(values),
                grad = lambda values: -1 * self.grad(values)
            )

    def __truediv__(self, other):
        return self.__mul__(other ** -1)
    
    def __rtruediv__(self, other):
        return (self.__truediv__(other)) ** -1

    @ staticmethod 
    def exp(other):
        if isinstance(other, Variable): # e^a variable
            return Variable(
                name = None,
                evaluate = lambda values : pow(math.e, other.evaluate(values)),
                grad = lambda values : pow(math.e, other.evaluate(values)) * other.grad
            )
        else: # e ^ c constant
            return Variable(
                name = None, 
                evaluate = lambda values : pow(math.e, other),
                grad =lambda values : 0
            )
            
    @ staticmethod
    def ln(other):
        if isinstance(other, Variable): #ln(a variable)
            return Variable(
                name = None,
                evaluate = lambda values : math.ln(other.evaluate(values)),
                grad = lambda values : pow(other.evaluate(values), -1) * other.grad(values)
            )