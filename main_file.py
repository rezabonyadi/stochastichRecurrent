import numpy as np
import SRNN
import matplotlib.pyplot as plt
import tkinter
from statsmodels.tsa.ar_model import AR

degree = 3
samples = 1000
instances = 200
rand_info = {"type": "rand", "var": .1, "mean": .5}

my_srnn = SRNN.SRNN(degree, rand_info, [0.2, 0.1, 0.8, 0.4])
# print(my_srnn.coefs)
init_points = np.random.rand(degree)

s = my_srnn.generate_sequence(init_points, samples)

ar_model = AR(s).fit(degree)
print(2*np.roll(ar_model.params, -1))
print(my_srnn.coefs)

c = np.zeros((instances, samples))
for i in range(instances):
    c[i, :] = (my_srnn.generate_sequence(init_points, samples))

# print((c[:, degree])) # The next sample right after initial points

for i in range(10):
    plt.plot(c[i, 0:10])

plt.show()

for j in range(3):
    print(np.mean(c[:, degree + j]), np.var(c[:, degree + j])) # The mean of next sample right after the initial points
# # for i in range(instances):
# #     plt.plot(c[i, 0:degree + 10])
#
plt.figure()
plt.hist([c[:, degree], c[:, degree + 1]])
plt.show()
#
# plt.figure()
# plt.hist(c[:, degree + 1])
#
# plt.figure()
# plt.hist(c[:, degree + 2])
#
# plt.show()

#
# seq1 = my_srnn.generate_sequence(init_points, samples)
# seq2 = my_srnn.generate_sequence(init_points, samples)
# seq3 = my_srnn.generate_sequence(init_points, samples)
#
# seq1 = seq1[len(seq1)-200:len(seq1)]
# seq2 = seq2[len(seq2)-200:len(seq2)]
# seq3 = seq3[len(seq3)-200:len(seq3)]
# # plt.switch_backend('QT4Agg') #TkAgg (instead Qt4Agg)
#
# plt.plot(np.arange(len(seq1)), seq1, '-ro', np.arange(len(seq2)), seq2, '-bo',
#          np.arange(len(seq3)), seq3, '-ko')
# mng = plt.get_current_fig_manager()
# # mng.full_screen_toggle()
# # plt.tight_layout()
#
# fft_seq1 = np.absolute(np.fft.fft(seq1 - np.mean(seq1)))
# fft_seq2 = np.absolute(np.fft.fft(seq2 - np.mean(seq2)))
# fft_seq3 = np.absolute(np.fft.fft(seq3 - np.mean(seq3)))
#
# fft_seq1 /= fft_seq1.sum()
# fft_seq2 /= fft_seq2.sum()
# fft_seq3 /= fft_seq3.sum()
#
# plt.figure()
# plt.plot(np.arange(len(fft_seq1)), fft_seq1, '-ro', np.arange(len(fft_seq2)), fft_seq2, '-bo',
#          np.arange(len(fft_seq3)), fft_seq3, '-ko')
#
# plt.show()
# i = 0

