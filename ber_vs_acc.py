import numpy as np
import matplotlib.pyplot as plt


ber = np.flip(np.load('ber_results.npy'), axis=1) # EsEp by SNR
print(ber.shape)
acc = np.concatenate((np.expand_dims(np.load('None.npy'), axis=0),
                      np.expand_dims(np.load('FGSM.npy'), axis=0),
                      np.expand_dims(np.load('BL-FGSM.npy'), axis=0),
                      np.expand_dims(np.load('single.npy'), axis=0),
                      np.expand_dims(np.load('BL-Single.npy'), axis=0)), axis=0)
print(acc.shape)
plt.semilogx(ber[0], acc[0], linestyle='--', marker='o', color='pink')
plt.semilogx(ber[1], acc[1], linestyle='-.', marker='^', color='tab:orange')
plt.semilogx(ber[2], acc[2], linestyle='-', marker='s', color='b')
plt.semilogx(ber[3], acc[3], linestyle=':', marker='*', color='red')
plt.semilogx(ber[4], acc[4], linestyle='-', marker='s', color='m')
plt.xlabel("BER")
plt.ylabel("Classification Accuracy")
plt.grid()
# plt.ylim([0, 1])
# plt.xlim([5e-2, 1.])
# plt.legend([r'$\infty$ dB','20 dB','16 dB','12 dB','8 dB','4 dB','0 dB'], title='Es/Ep')
plt.legend(['No Perturbation','FGSM','BL FGSM','Single','BL Single'])
plt.show()
