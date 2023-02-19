import random
class Model:
    def __init__(self, data, alpha):
        self.data = data #list of tuples consisting of x & y coordinates
        self.a0 = random.uniform(0,100) #random float from 1 to 100, the b in mx + b
        self.a1 = random.uniform(1,100) #the m in mx + b
        self.alpha = alpha #small coefficient to the gradient of the 
        self.data_size = len(self.data)
    def fit_function(self,x):
        return self.a0 + (self.a1)*x
    def cost(self):
        return sum([(self.data[i][1]) - self.fit_function(self.data[i][0])**2 for i in range(self.data_size)])
    def gradient(self):
        return (sum([(-2*((self.data[j])[1]) + 2*((self.a0) + ((self.data[j])[0])*(self.a1))) for j in range(self.data_size)]),sum([((-2*(self.data[k])[1])*((self.data[k])[0]) + 2*((self.data[k])[0])*((self.a0) + ((self.data[k])[0])*self.a1)) for k in range(self.data_size)]))
    def gradient_descent(self):
        grad = self.gradient()
        self.a0 = self.a0 - self.alpha * grad[0]
        self.a1 = self.a1 - self.alpha * grad[1]
        return (self.a0,self.a1)
    def train(self,reps):
        for r in range(reps):
            self.gradient_descent()
