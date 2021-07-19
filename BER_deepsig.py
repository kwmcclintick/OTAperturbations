'''
Performs an over the air attack on an eavesdropping VT-CNN2 (O'Shea) modulation classifier by injecting the
RML2016.10a data set with:
1) FGSM attacks (benchmark)
2) single pixel attacks (proposed)
in an effort to decrease BER for equal decreases in classification accuracy of the adversarial classifier
'''




# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import numpy as np
import matplotlib.pyplot as plt


# create bits
bits = np.load('bits.npy')

# EsEps = np.array([-np.infty, -20, -16, -12, -8, -4, 0])
EsEps = np.array([0,-10])
snrs = range(-20,20,2)
attacks = ['fgsm','bl-fgsm','single','bl-single','2-bl-single','3-bl-single','boost']
# attacks = ['bl-fgsm','_boostbl-fgsm','bl-single','_boostbl-single']
# ber_result = np.zeros(shape=(len(EsEps),len(snrs)))
ber_result = np.zeros(shape=(len(attacks), len(EsEps), len(snrs)))

for attack in range(len(attacks)):
    print(attack)
    for EsEp in range(len(EsEps)):
        for snr in range(len(snrs)):
            if EsEp == 0:
                received = np.load('received'+str(attacks[attack])+str(EsEps[EsEp])+str(snrs[snr])+'.npy')
            else:
                received = np.load('received'+str(attacks[attack])+str(int(EsEps[EsEp]))+str(snrs[snr])+'.npy')

            # plt.plot(received)
            # plt.show()
            # clip convolution tails
            # downsample
            min_ber = np.infty
            for idx in range(24):
                decimated = received[idx:idx+10000]
                decimated = decimated[::2]
                # threshold
                rec_bits = []
                for i in range(len(decimated)):
                    if np.real(decimated[i]) < 0 and np.imag(decimated[i]) > 0:  # 01
                        rec_bits.append(0)
                        rec_bits.append(1)
                    if np.real(decimated[i]) > 0 and np.imag(decimated[i]) < 0:  # 10
                        rec_bits.append(1)
                        rec_bits.append(0)
                    if np.real(decimated[i]) > 0 and np.imag(decimated[i]) > 0:  # 00
                        rec_bits.append(0)
                        rec_bits.append(0)
                    if np.real(decimated[i]) < 0 and np.imag(decimated[i]) < 0:  # 11
                        rec_bits.append(1)
                        rec_bits.append(1)

                # computer bit error rate
                ber = np.sum(np.abs(bits-rec_bits))/len(bits)
                #print(ber)
                if ber < min_ber:
                    min_ber = ber
            #print(min_ber)
            # ber_result[EsEp][snr] = min_ber
            ber_result[attack][EsEp][snr] = min_ber



# # for compare plot
# control = []
# for snr in range(len(snrs)):
#     received = np.load('received' + 'boost' + str(0) + str(snrs[snr]) + '.npy')
#     min_ber = np.infty
#     for idx in range(24):
#         decimated = received[idx:idx + 10000]
#         decimated = decimated[::2]
#         # threshold
#         rec_bits = []
#         for i in range(len(decimated)):
#             if np.real(decimated[i]) < 0 and np.imag(decimated[i]) > 0:  # 01
#                 rec_bits.append(0)
#                 rec_bits.append(1)
#             if np.real(decimated[i]) > 0 and np.imag(decimated[i]) < 0:  # 10
#                 rec_bits.append(1)
#                 rec_bits.append(0)
#             if np.real(decimated[i]) > 0 and np.imag(decimated[i]) > 0:  # 00
#                 rec_bits.append(0)
#                 rec_bits.append(0)
#             if np.real(decimated[i]) < 0 and np.imag(decimated[i]) < 0:  # 11
#                 rec_bits.append(1)
#                 rec_bits.append(1)
#
#         # computer bit error rate
#         ber = np.sum(np.abs(bits - rec_bits)) / len(bits)  # remove 1- ?
#         if ber < min_ber:
#             min_ber = ber
#     control.append(min_ber)

np.save('ber_results', ber_result)

ber_result = np.load('ber_results.npy')



# computed using Matlab's berawgn((-20:2:18)+3,'psk',4,'nondiff')
theory0 = [0.420832979535499,
            0.400718927292596,
            0.375772437287866,
            0.345101521134146,
            0.307910470715079,
            0.263789505256266,
            0.213228018357620,
            0.158368318809598,
            0.103759095953406,
            0.0562819519765415,
            0.0228784075610853,
            0.00595386714777866,
            0.000772674815378444,
            3.36272284196176e-05,
            2.61306795357521e-07,
            1.33293101753005e-10,
            9.12395736262818e-16,
            6.75896977065478e-24,
            1.00107397357086e-36,
            5.29969722887052e-57]

theory10 = [0.441043549731138,
            0.425936631294734,
            0.407074232722028,
            0.383635955449055,
            0.354733147827677,
            0.319519959185737,
            0.277431643894643,
            0.228625582368683,
            0.174672384668755,
            0.119362592358735,
            0.0690053687843298,
            0.0309306798819288,
            0.00936714146394295,
            0.00154093217875996,
            9.73755174612455e-05,
            1.36325232807776e-06,
            1.76447921215662e-09,
            5.27575447875283e-14,
            4.03963421898187e-21,
            2.42260300497505e-32]


plt.semilogy(snrs, theory0, linestyle='-', color='black')
plt.semilogy(snrs, theory10, linestyle='--', color='black')
plt.semilogy(snrs, ber_result[0,0], linestyle='-', marker='^', color='tab:orange')
plt.semilogy(snrs, ber_result[0,1], linestyle='--', marker='^', color='tab:orange')
plt.semilogy(snrs, ber_result[1,0], linestyle='-', marker='s', color='b')
plt.semilogy(snrs, ber_result[1,1], linestyle='--', marker='s', color='b')
plt.semilogy(snrs, ber_result[2,0], linestyle='-', marker='*', color='red')
plt.semilogy(snrs, ber_result[2,1], linestyle='--', marker='*', color='red')
plt.semilogy(snrs, ber_result[3,0], linestyle='-', marker='s', color='m')
plt.semilogy(snrs, ber_result[3,1], linestyle='--', marker='s', color='m')
plt.semilogy(snrs, ber_result[4,0], linestyle='-', marker='D', color='m')
plt.semilogy(snrs, ber_result[4,1], linestyle='--', marker='D', color='m')
plt.semilogy(snrs, ber_result[5,0], linestyle='-', marker='|', color='m')
plt.semilogy(snrs, ber_result[5,1], linestyle='--', marker='|', color='m')
plt.semilogy(snrs, ber_result[6,0], linestyle='-', marker='o', color='pink')
plt.semilogy(snrs, ber_result[6,1], linestyle='--', marker='o', color='pink')




# plt.legend(['Theory','No Perturbation','BL FGSM','Boosting BL FGSM','BL Single','Boosting BL Single'])
plt.legend(['0 dB Theory','10 dB Theory','0 dB FGSM','10 dB FGSM','0 dB BL FGSM','10 dB BL FGSM',
            '0 dB Single','10 dB Single','0 dB BL Single','10 dB BL Single','0 dB BL Double','10 dB BL Double',
            '0 dB BL Triple','10 dB BL Triple','0 dB None','10 dB None'], title='Es/Ep, Perturbation')

plt.grid()
plt.ylim([1e-4, 0.5])
plt.xlabel(r"$E_s/N_0$ (dB)")
plt.ylabel("BER")
plt.show()
