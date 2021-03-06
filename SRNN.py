import numpy as np


class SRNN():
    degree = 1
    rand_info = []
    coefs_means = [1, 1]

    def __init__(self, rand_info):
        self.degree = len(rand_info["means"])
        self.rand_info = rand_info

        # if init_coefs is None:
        #     self.initialize_coefs()
        # else:
        #     self.coefs = init_coefs

    def initialize_coefs(self):
        """
        Initialises the coefficients.

        :return:
        """
        # self.coefs = 1.0 * np.random.randn(self.degree + 1)/self.degree

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
            for j in range(0, self.degree - 1):
                c = c + self.get_rand(j)*seq[i - 1 - j]

            c += self.get_rand(self.degree - 1)
            seq[i] = c

        return seq

    def get_rand(self, ceof_indx):
        res = np.random.rand(1)

        if self.rand_info["dists"][ceof_indx] == "rand":
            a = self.rand_info["means"][ceof_indx] - np.sqrt(12.0 * self.rand_info["vars"][ceof_indx])/2.0
            b = self.rand_info["means"][ceof_indx] + np.sqrt(12.0 * self.rand_info["vars"][ceof_indx])/2.0
            res = (np.random.rand(1) * (b - a)) + a
        if self.rand_info["dists"][ceof_indx] == "randn":
            res = (np.random.randn(1) * np.sqrt(self.rand_info["vars"][ceof_indx]) + self.rand_info["means"][ceof_indx])

        return res
