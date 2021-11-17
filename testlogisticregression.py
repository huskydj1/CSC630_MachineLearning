from logisticregression import LogisticRegression
import numpy as np

X = np.array([[1,1,1],[-1,-1,-1],[-2,-2,-2],[-2,-2,0],[0,-2,-2],[3,0,3]])
y = np.array([1,0,0,0,0,1])

model = LogisticRegression()
model.fit(
    X = X,
    y = y,
    learning_rate = 0.1,
    num_epochs = 20
)

X_predict = np.array([[1,1,1],[3,0,2],[0,0,0],[-1,0,-1]])
print(model.predict(X_predict))