import numpy as np
import SRNN
import matplotlib.pyplot as plt
import tkinter

samples = 1000
degree = 10
my_srnn = SRNN.SRNN(degree, "rand")
print(my_srnn.coefs)
init_points = np.random.rand(degree)

c = np.zeros((1000, degree + 1))
for i in range(1000):
    c[i, :] = (my_srnn.generate_sequence(init_points, degree + 1))
print((c[:, degree]))

print(np.mean(c[:, degree]))
print(np.var(c[:, degree]))

seq1 = my_srnn.generate_sequence(init_points, samples)
seq2 = my_srnn.generate_sequence(init_points, samples)
seq3 = my_srnn.generate_sequence(init_points, samples)

seq1 = seq1[len(seq1)-200:len(seq1)]
seq2 = seq2[len(seq2)-200:len(seq2)]
seq3 = seq3[len(seq3)-200:len(seq3)]
# plt.switch_backend('QT4Agg') #TkAgg (instead Qt4Agg)

plt.plot(np.arange(len(seq1)), seq1, '-ro', np.arange(len(seq2)), seq2, '-bo',
         np.arange(len(seq3)), seq3, '-ko')
mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
# plt.tight_layout()

fft_seq1 = np.absolute(np.fft.fft(seq1 - np.mean(seq1)))
fft_seq2 = np.absolute(np.fft.fft(seq2 - np.mean(seq2)))
fft_seq3 = np.absolute(np.fft.fft(seq3 - np.mean(seq3)))

fft_seq1 /= fft_seq1.sum()
fft_seq2 /= fft_seq2.sum()
fft_seq3 /= fft_seq3.sum()

plt.figure()
plt.plot(np.arange(len(fft_seq1)), fft_seq1, '-ro', np.arange(len(fft_seq2)), fft_seq2, '-bo',
         np.arange(len(fft_seq3)), fft_seq3, '-ko')

plt.show()
i = 0

