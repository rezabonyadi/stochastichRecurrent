import numpy as np


class SRNN():
    degree = 1
    rand_type = "rand"
    coefs = [1, 1]

    def __init__(self, degree, rand_type, init_coefs=None):
        self.degree = degree
        self.rand_type = rand_type
        if init_coefs is None:
            self.initialize_coefs()
        else:
            self.coefs = init_coefs

    def initialize_coefs(self):
        """
        Initialises the coefficients.

        :return:
        """
        self.coefs = np.random.randn(self.degree + 1)/self.degree

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
                c = c + self.coefs[j]*np.random.rand(1)*seq[i - self.degree + j]

            c += self.coefs[self.degree]
            seq[i] = c

        return seq

