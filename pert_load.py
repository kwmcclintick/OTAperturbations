import pickle, keras
import numpy as np
import matplotlib.pyplot as plt
def to_onehot(yy):
    hotshot_list = np.array(list(yy))
    yy1 = np.zeros([len(hotshot_list), np.max(hotshot_list)+1])
    yy1[np.arange(len(hotshot_list)), hotshot_list] = 1
    return yy1

# while mods[yt[idx]] != 'BPSK':
#     idx += 1
#     while snrs[ysnr[idx]] != 18:
#         idx += 1

# print(X_train.shape)
# # pert = np.reshape(X_train, (2,500*128))
# # pert = X_train[idx]
# print(np.sum(np.abs(pert)))
#
# print(Y_train[idx])  # this doesn't align correctly [0 ...] is [0 1 0 ...] for all classes
# print(mods[yt[idx]])
# print(snrs[ysnr[idx]])

# Load the dataset ...
#  You will need to seperately download or generate this file
with (open("RML2016.10a_pow_control.dat",'rb')) as openfile:
    Xd = pickle.load(openfile, encoding="latin1")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
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
X_test =  X[test_idx]

Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
yt = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
ysnr = list(map(lambda x: snrs.index(lbl[x][1]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
X_train_con = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# control = X_train[idx]








# Load the dataset ...
#  You will need to seperately download or generate this file
with (open("RML2016.10a_rml_nochannel.dat",'rb')) as openfile:
    Xd = pickle.load(openfile, encoding="latin1")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

X = []
lbl = []
for mod in mods:
    if mod=='BPSK':
        for snr in snrs:
          if snr==0:
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
X_test =  X[test_idx]
def to_onehot(yy):
    hotshot_list = np.array(list(yy))
    yy1 = np.zeros([len(hotshot_list), np.max(hotshot_list)+1])
    yy1[np.arange(len(hotshot_list)), hotshot_list] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
yt = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
ysnr = list(map(lambda x: snrs.index(lbl[x][1]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
X_train_nochan = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


# Load the dataset ...
#  You will need to seperately download or generate this file
with (open("RML2016.10a_rml_channel.dat",'rb')) as openfile:
    Xd = pickle.load(openfile, encoding="latin1")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

X = []
lbl = []
for mod in mods:
    if mod=='BPSK':
        for snr in snrs:
          if snr==0:
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
X_test =  X[test_idx]

Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
yt = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
ysnr = list(map(lambda x: snrs.index(lbl[x][1]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

idx = 0  # search



# Psig = np.sum(np.power(control,2))
# noise = pert - control
# Pn = np.sum(np.power(noise,2))
# print(10*np.log10(Psig/Pn))

avg_control = []
avg_rml = []
snr_mis = 0
for i in range(len(X_train)):
    control = X_train_con[i]
    rml = X_train[i]
    no_chan = X_train_nochan[i]
    avg_rml.append(rml)

    snr_mis += 10*np.log10(np.sum(np.abs(rml)**2)/np.sum(np.abs(no_chan)**2))

    control = control[0,:] + 1j*control[1,:]
    Psig = np.sum(np.abs(control)**2)
    Pn = Psig / np.power(10.0,0.0/10.0)

    noise = np.sqrt(Pn/2)*(np.random.normal(0,1,control.shape) + 1j*np.random.normal(0,1,control.shape))
    control += noise
    Pn = np.sum(np.abs(noise)**2)
    Psig = np.sum(np.abs(control)**2)
    # print(10*np.log10(Psig/Pn))
    avg_control.append(control)

print(snr_mis / i)

avg_rml = np.array(np.mean(avg_rml, axis=0))
avg_rml = avg_rml[0,:] + 1j*avg_rml[1,:]
avg_control = np.array(np.mean(avg_control, axis=0))
# avg_control = avg_control[0,:] + 1j*avg_control[1,:]
print(avg_control.shape)

plt.psd(avg_control, Fs=1, scale_by_freq=False)
plt.psd(avg_rml, Fs=1, scale_by_freq=False)
plt.xlabel('Normalized Frequency')
plt.legend(['Average Obtained 0 dB SNR','Average RML2016.10a 0 dB SNR'])
plt.show()

# plt.plot(control[0,:,0], 'b', marker='v')
# plt.plot(control[1,:,0],'r',marker='o')
# plt.plot(pert[0,:], pert[1,:], linewidth=0.05)
# plt.xlim([-.01, .01])
# plt.ylim([-.01, .01])
# plt.plot(pert[0,:], 'b--')
# plt.plot(pert[1,:], 'r--')
# plt.xlabel('Samples')
# plt.ylabel('Amplitude')
# # plt.xlabel('I')
# # plt.ylabel('Q')
# plt.legend(['I','Q','Adversarial I','Adversarial Q'])
# # plt.legend(['I','Q'])
# plt.grid()
# plt.show()

