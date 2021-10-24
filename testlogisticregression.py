from logisticregression import LogisticRegression
import numpy as np

X, y = np.array([[0], [1], [2], [4], [5], [6]]), np.array([0, 0, 0, 0, 0, 0])
model = LogisticRegression()
model.fit(X, y)