import random
class PolynomialFittingModel:
    def __init__(self, data, alpha, degree):
        self.degree = degree + 1
        self.data = data
        self.alpha = alpha
        self.coefficients = [random.uniform(0,100) for _ in range(self.degree)]
        self.data_size = len(data)
    def fit_function(self,x):
        return sum([(self.coefficients[i])*(x**i) for i in range(self.degree)])
    def partial_derivative(self,k):
        return sum([(((-2*(self.data[i])[1])*((self.data[i])[0])**k)+2*(((self.data[i])[0])**k)*sum([self.fit_function(self.data[i][0])])) for i in range(self.data_size)])
    def gradient_descent(self):
        for j in range(self.degree):
            self.coefficients[j] = self.coefficients[j] - (self.alpha)*self.partial_derivative(j)
        return self.coefficients
    def train(self,reps):
        for _ in range(reps):
            self.gradient_descent()