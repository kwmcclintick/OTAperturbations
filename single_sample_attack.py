'''
Performs an over the air attack on an eavesdropping VT-CNN2 (O'Shea) modulation classifier by injecting the
RML2016.10a data set with:
1) FGSM attacks (benchmark)
2) single pixel attacks (proposed)
in an effort to decrease BER for equal decreases in classification accuracy of the adversarial classifier
'''

# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import os, sys
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid()

# Load the dataset ...
#  You will need to seperately download or generate this file
with (open("RML2021.10a.varyP.FGSM_98.dat",'rb')) as openfile:
    Xd = pickle.load(openfile, encoding="latin1")

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

X = []
lbl = []
for mod in mods:
    if mod=='QPSK':
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]

def to_onehot(yy):
    hotshot_list = np.array(list(yy))
    yy1 = np.zeros([len(hotshot_list), np.max(hotshot_list)+1])
    yy1[np.arange(len(hotshot_list)), hotshot_list] = 1
    return yy1

Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

Y_train = np.zeros((len(Y_train),11))
Y_test = np.zeros((len(Y_test),11))
Y_train[:,9] = 1.  # 9 for qpsk
Y_test[:,9] = 1.

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
in_shp = list(X_train.shape[1:])
classes = mods

model = keras.models.load_model('VT-CNN2')  # trained model
class_key = np.array(['AM-SSB','AM-DSB','WBFM','CPFSK','GFSK','QAM64','QAM16','PAM4','8PSK','QPSK','BPSK'])
# Plot confusion matrix
import tensorflow as tf
acc = np.zeros(shape=(len(snrs)))
# Compute performance over each SNR
snr_count = 0  # snr counter
class_scores = []
snr_list = []
class_list = []
for snr in snrs:
    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    score_delta = test_Y_i_hat[:,9] - np.max(np.delete(test_Y_i_hat, 9, axis=1), axis=1)
    # class_scores.extend(test_Y_i_hat.flatten())
    class_scores.extend(score_delta)
    # class_list.extend(np.tile(class_key,len(test_Y_i_hat)))
    # snr_list.extend(np.repeat(snr, len(test_Y_i_hat)*len(test_Y_i_hat[0])))
    snr_list.extend(np.repeat(snr, len(test_Y_i_hat)))

    acc[snr_count] = np.sum(np.argmax(test_Y_i_hat,axis=1)==np.argmax(test_Y_i,axis=1)) / len(test_Y_i_hat)
    # conf = np.zeros([len(classes), len(classes)])
    # confnorm = np.zeros([len(classes), len(classes)])
    # for i in range(test_X_i.shape[0]):
    #     j = list(test_Y_i[i, :]).index(1)
    #     k = int(np.argmax(test_Y_i_hat[i, :]))
    #     conf[j, k] = conf[j, k] + 1
    # for i in range(len(classes)):
    #     confnorm[i, :] = conf[i, :] / (np.sum(conf[i, :]) + sys.float_info.min)
    # if snr==18:
    #     plt.figure()
    #     plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))
    #     plt.show()
    # cor = np.sum(np.diag(conf))
    # ncor = np.sum(conf) - cor
    # # print("Overall Accuracy: ", cor / (cor + ncor))
    # acc[snr_count] = 1.0 * cor / (cor + ncor)
    snr_count += 1


import seaborn
import pandas
snr_list = np.array(snr_list)
class_scores = np.array(class_scores)
class_list = np.array(class_list)


# # DF = pandas.DataFrame({'SNR':snr_list,'Class_Scores':class_scores,'class':class_list})
# # seaborn.lineplot(x='SNR',y='Class_Scores',data=DF,hue='class',err_style="bars", ci=0)
# DF = pandas.DataFrame({r'$E_p/E_s$':snr_list,'Class Scores':class_scores,'class':class_list})
# seaborn.lineplot(x=r'$E_p/E_s$',y='Class Scores',data=DF,hue='class',err_style="bars", ci=0)
# plt.title(r'SNR$=10$ dB QPSK Signals Under Attack by FGSM Perturbations')
# plt.tight_layout()
# plt.grid()
# plt.show()


print('saving into data frame...')
# DF = pandas.DataFrame({r"$E_s/N_0$ (dB)": snr_list, r'$\Delta_{logits}$': class_scores})
DF = pandas.DataFrame({r"$E_p/E_s$ (dB)": snr_list, r'$\Delta_{logits}$': class_scores})
# seaborn.lineplot(x='SNR',y='Class_Scores',data=DF)
print('Loading gradients...')
grads = np.load('awgn_grads.npy')
grads = np.concatenate((np.expand_dims(np.real(grads), axis=-1), np.expand_dims(np.imag(grads), axis=-1)), axis=-1).flatten()
print('Loading SNR list...')
snr_grads = np.load('grads_snr.npy')
snr_grads = np.tile(snr_grads, 2)
print('creating data frame...')
# DF_g = pandas.DataFrame({r"$E_s/N_0$ (dB)": snr_grads, r'$\nabla_{Rx}$': np.real(grads)})
DF_g = pandas.DataFrame({r"$E_p/E_s$ (dB)": snr_grads, r'$\nabla_{Rx}$': np.real(grads)})
print('plotting...')

# ax = seaborn.lineplot(x=r"$E_s/N_0$ (dB)", y=r'$\Delta_{logits}$', data=DF)
ax = seaborn.lineplot(x=r"$E_p/E_s$ (dB)", y=r'$\Delta_{logits}$', data=DF)
ax.set_ylabel("Class Score Difference with Truth", color='tab:blue')
ax.tick_params(axis='y', labelcolor='tab:blue')
ax.plot(snrs, np.zeros(shape=(len(snrs))), linestyle='--', color='tab:blue')

ax2 = ax.twinx()
ax2.set_ylabel(r'$\nabla_{Rx}$', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
# seaborn.lineplot(x=r"$E_s/N_0$ (dB)", y=r'$\nabla_{Rx}$', ax=ax2, data=DF_g, color='tab:orange')
seaborn.lineplot(x=r"$E_p/E_s$ (dB)", y=r'$\nabla_{Rx}$', ax=ax2, data=DF_g, color='tab:orange')
plt.grid()
# plt.title(r'QPSK Signals Under Attack by $E_s/E_p=10$ dB FGSM Perturbations')
plt.title(r'SNR$=10$ dB QPSK Signals Under Attack by FGSM Perturbations')
plt.tight_layout()
plt.show()

# np.save('FGSM.npy', acc)

# # Plot accuracy curve
# plt.plot(snrs, np.load('None.npy'), linestyle='-', marker='o', color='pink')
# plt.plot(snrs, np.load('None-10.npy'), linestyle='--', marker='o', color='pink')
# plt.plot(snrs, np.load('FGSM.npy'), linestyle='-', marker='^', color='tab:orange')
# plt.plot(snrs, np.load('FGSM-10.npy'), linestyle='--', marker='^', color='tab:orange')
# plt.plot(snrs, np.load('BL-FGSM.npy'), linestyle='-', marker='s', color='b')
# # plt.plot(snrs, np.load('boost-BL-FGSM.npy'), linestyle='--', marker='D', color='b')
# plt.plot(snrs, np.load('BL-FGSM-10.npy'), linestyle='--', marker='s', color='b')
# plt.plot(snrs, np.load('single.npy'), linestyle='-', marker='*', color='red')
# plt.plot(snrs, np.load('single-10.npy'), linestyle='--', marker='*', color='red')
# plt.plot(snrs, np.load('BL-Single.npy'), linestyle='-', marker='s', color='m')
# # plt.plot(snrs, np.load('boost-BL-Single.npy'), linestyle='--', marker='D', color='m')
# plt.plot(snrs, np.load('BL-Single-10.npy'), linestyle='--', marker='s', color='m')
# plt.plot(snrs, np.load('2-BL-Single.npy'), linestyle='-', marker='D', color='m')
# plt.plot(snrs, np.load('2-BL-Single-10.npy'), linestyle='--', marker='D', color='m')
# plt.plot(snrs, np.load('3-BL-Single.npy'), linestyle='-', marker='|', color='m')
# plt.plot(snrs, np.load('3-BL-Single-10.npy'), linestyle='--', marker='|', color='m')
#
# plt.xlabel(r"$E_s/N_0$ (dB)")
# # plt.xlabel('Max LO Offset (PPM)')
# # plt.xlabel('Max Timing Offset Per Sample (Samples)')
# plt.ylabel("Classification Accuracy")
# plt.grid()
# plt.ylim([0, 1.05])
# plt.legend(['0 dB None','10 dB None','0 dB FGSM','10 dB FGSM','0 dB BL FGSM','10 dB BL FGSM',
#             '0 dB Single','10 dB Single','0 dB BL Single','10 dB BL Single','0 dB BL Double','10 dB BL Double',
#             '0 dB BL Triple','10 dB BL Triple'], title='Es/Ep, Perturbation')
# # plt.legend(['None','BL FGSM','Boosting BL FGSM','BL Single','Boosting BL Single'])
# plt.show()




