import numpy as np


class SRNN():
    degree = 1
    rand_info = []
    coefs = [1, 1]

    def __init__(self, degree, rand_info, init_coefs=None):
        self.degree = degree
        self.rand_info = rand_info
        if init_coefs is None:
            self.initialize_coefs()
        else:
            self.coefs = init_coefs

    def initialize_coefs(self):
        """
        Initialises the coefficients.

        :return:
        """
        self.coefs = 1.0 * np.random.randn(self.degree + 1)/self.degree

    def generate_sequence(self, init_points, steps):
        """
        Generates a sequence, given the initial points and the number of samples in the sequence.

        :param init_points:
        :param steps:
        :return:
        """
        seq = np.zeros(steps)
        seq[0:len(init_points)] = init_points

        for i in range(len(init_points), steps):
            c = 0
            for j in range(0, self.degree):
                c = c + self.coefs[j]*self.get_rand()*seq[i - 1 - j]

            c += self.coefs[self.degree]*self.get_rand()
            seq[i] = c

        return seq

    def get_rand(self):
        res = np.random.rand(1)

        if self.rand_info["type"] == "rand":
            a = self.rand_info["mean"] - np.sqrt(12.0 * self.rand_info["var"])/2.0
            b = self.rand_info["mean"] + np.sqrt(12.0 * self.rand_info["var"])/2.0
            res = (np.random.rand(1) * (b - a)) + a

        return res
