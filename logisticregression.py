from variable import Variable
import numpy as np
import math

# TODO: SEPARATE VARIABLE PARAMETERS (INITALIZE Y_HAT ONCE) WITH VALUE OF PARAMETERS


class LogisticRegression():

    def fit(self, X, y):
        # Store shape
        self.num_samples, self.num_features = X.shape
        assert(len(y)==self.num_samples)

        # Store parameters
        self.m_values, self.b_value = self.initializeParameters()

        # Gradient Descent
        for iter in range(0, 100):
            pass
    
    def initializeParameters(self): # TODO: COMPLETELY RANDOMIFY EVERYTHING BY ADDING SEED TO RANDOM
        m_values = [math.random() - 0.5 for i in range(self.num_features)]
        b_value = math.random() - 0.5

        return m_values, b_value


    def generate_yhat(self, x):
        y_hat = sum([Variable(name = f"m{i}") * x[i] for i in range(self.num_features)]) + Variable(name = "b")
        return y_hat

    def evaluate(self, x):
        # Get Variable y_hat
        y_hat = self.generate_yhat(x)

        # Dictinoify Parameters
        parameters = {f"m{i}" : self.m_values[i] for i in range(self.num_features)}
        parameters["b"] = self.b_value

        # Evaluate
        return y_hat(**parameters)


    def gradient(self, x):
        # Get Variable y_hat
        y_hat = self.generate_yhat(x)

        # Dictinoify Parameters
        parameters = {f"m{i}" : self.m_values[i] for i in range(self.num_features)}
        parameters["b"] = self.b_value

        # Take gradient
        grad_vector = y_hat.grad(**parameters)

        return grad_vector