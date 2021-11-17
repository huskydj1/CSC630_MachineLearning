from variable import Variable
import numpy as np
import math
import random

class LogisticRegression():

    @staticmethod
    def dictionify(m_var, b_var, m, b):
        dict = {}
        for i in range(len(m_var)):
            dict[m_var[i].name] = m[i]
        dict[b_var.name] = b
        return dict

    @staticmethod
    def evaluate_model(X, y, m, b):
        num_samples, num_features = X.shape
        assert(len(y)==num_samples)

        scores = []
        for i in range(num_samples):
            y_hat = 1/(1 + math.exp(-1 * (
                sum([X[i, j] * m[j] for j in range(num_features)], b)
            )))
            y_prediction = 1 if y_hat >= 0.5 else 0

            scores.append(y[i]==y_prediction)
        
        return sum(scores)/len(scores)

    def predict(self, X):
        num_samples, num_features = X.shape

        predictions = []
        for i in range(num_samples):
            y_hat = 1/(1 + math.exp(-1 * (
                sum([X[i, j] * self.m[j] for j in range(num_features)], self.b)
            )))
            y_prediction = 1 if y_hat >= 0.5 else 0

            predictions.append(y_prediction)
        
        return predictions

    def fit(self, X, y, learning_rate = 1, num_epochs = 20, printeverynthepoch = 0):
        # Reset Variable's Computational Graph
        Variable.resetIndependentVariables()

        # Store shape
        self.num_samples, self.num_features = X.shape
        assert(len(y)==self.num_samples)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # INITALIZE MODEL

        # Create m and b Variable Instances
        m_var = [Variable(name = f"m_var_{i}") for i in range(self.num_features)]
        b_var = Variable(name = "zzz_b")

        # Create y_hat Variable Instance
        y_hat_var = []
        for i in range(self.num_samples):
            y_hat_var_i = 1/(1 + Variable.exp(-1 * (
                sum([m_var[j] * X[i, j] for j in range(self.num_features)], b_var)
            )))
            y_hat_var.append(y_hat_var_i)

        # Create Cost Variable Instance
        cost = -1 * sum([(
            y[i] * Variable.log(y_hat_var[i]) + (1 - y[i]) * Variable.log(1 - y_hat_var[i])
        ) for i in range(self.num_samples)])
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # Generate Random Values of m and b
        m = [random.random() - 0.50 for i in range(self.num_features)]
        b = random.random() - 0.50

        # Training Loop
        print("Training in Progress")
        for iter in range(num_epochs):
            parameters = LogisticRegression.dictionify(
                m_var = m_var, 
                b_var = b_var, 
                m = m, 
                b = b
            )

            # Get Cost
            cost_value = cost(**parameters)

            # Get Score on Training Dataset
            training_score = LogisticRegression.evaluate_model(X, y, m, b)

            if printeverynthepoch > 0 and iter%printeverynthepoch==0:
                print("ITER:", iter)
                print("COST:", cost_value)
                print("TRAINING SCORE:", training_score)


            # Take Gradient
            cost_grad = cost.gradient(**parameters)

            # Update Parameters
            b -= learning_rate * cost_grad[-1]
            for i in range(self.num_features):
                m[i] -= learning_rate * cost_grad[i]

        # Store Weights
        self.m = m
        self.b = b
        