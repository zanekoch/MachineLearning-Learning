import numpy as np

#sigmoid defs
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))
def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    #y is the output of feed forward
    def __init__(self,x,y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        #y is calculated by y^ = σ(W2*σ(W1*x + b1) + b2)
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        #sum of squares loss fxn, which we seek to minimize
        # = Σ(y - y^)^2
        #difference between each predicted value and each actual value
        #Gradient Descent: take derivative of loss function to find direction of local minima
        # d(loss(y,y^))/dW = 2(y - y^) * (Wx + b(1- (wX + B)) * x
        #different loss calculation for each set of weights
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        #update weights with these changes
        self.weights1 += d_weights1
        self.weights2 += d_weights2

def main():
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X, y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()
    print(nn.output)


if __name__ == '__main__':
    main()
