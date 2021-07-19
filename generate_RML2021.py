#!/usr/bin/env python
from transmitters import transmitters
from source_alphabet import source_alphabet
import analyze_stats
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, cPickle, gzip
import random
from scipy.stats import truncnorm
import scipy.fftpack
import tensorflow as tf
import os, sys
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from tensorflow.keras.models import load_model
import warnings
import commpy
from gnuradio import gr, blocks, digital, analog, filter
from gnuradio.filter import firdes
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift
warnings.filterwarnings("ignore")
tf.compat.v1.enable_eager_execution()

def fft_plot(X,title):
        avg = []
	NFFT=512
        for val in X:
		X_val=fftshift(fft(val,NFFT))
		fs = 200e3
		fVals=np.arange(start = -NFFT/2,stop = NFFT/2)*fs/NFFT/fs
		psd = np.abs(X_val)**2
		dbm = 10*np.log10(psd/0.001)
        	avg.append(dbm)
        avg = np.array(avg)

	plt.plot(fVals,np.mean(avg,axis=0),label=title)
	plt.xlabel('Normalized Frequency')         
	plt.ylabel('PSD (dBm)')
	#plt.title(title)
	#plt.grid()

def single_fft_plot(X,title,color,marker):
	NFFT=512
	X_val=fftshift(fft(X,NFFT))
	fs = 200e3
	fVals=np.arange(start = -NFFT/2,stop = NFFT/2)*fs/NFFT/fs
	psd = np.abs(X_val)**2
	dbm = 10*np.log10(psd/0.001)
	plt.plot(fVals,dbm,label=title,color=color,linestyle='none',marker=marker)
	plt.xlabel('Normalized Frequency')         
	plt.ylabel('PSD (dBm)')
	plt.title(title)
	plt.grid()
        plt.show()


def time_plot(X,title):
	plt.plot(X, label=title)


fs = 200e3
sps = 2 # 8
nfilts = 1 # 32
ntaps = 8 # 11 * sps * nfilts
rrc_taps = filter.firdes.root_raised_cosine(
            1.0,          # gain
            200e3,          # sampling rate based on 32 filters in resampler
            200e3 / sps,             # symbol rate
            0.35, # excess bandwidth (roll-off factor)
            6*sps)
rrc_taps = np.convolve(rrc_taps,rrc_taps,'same')

'''
Generate dataset with dynamic channel model across range of SNRs
'''

dataset = {}
control_d = {}

model = load_model('VT-CNN2')

mods = np.array(['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','WBFM'])


EpEs_list = snr_vals = range(-20,100,2) # dB
print(EpEs_list)
p = 2
attack = 'fgsm'  # awgn, boost, reduce, fgsm, bl-fgsm, single, bl-single, p-bl-single

cfo_lims = np.array([1e-300, 50., 100., 150., 200., 250.])  # PPM
cfo_lims = np.array([1e-300])
t_offs = np.array([1e-300,1e-4,5e-4,1e-3,5e-3,1e-2])
t_offs = np.array([1e-300])

# Define low pass filter
# causes acc falloff at high SNR as low as (trying 0.9)
from scipy import signal
normalized_cutoff_freq = 0.65  # 0.25? for 8 sps, 0.x? for 2 sps
order = 11
numerator_coeffs, denominator_coeffs = scipy.signal.butter(order, normalized_cutoff_freq, 'low')
w, h = signal.freqz(numerator_coeffs, denominator_coeffs)
#plt.plot(w / np.pi, 20 * np.log10(abs(h)))
#plt.axis([0.08, 1, -60, 3])
#plt.xlabel('Normalized Frequency [radians / second]')
#plt.ylabel('Amplitude [dB]')
#plt.grid(which='both', axis='both')
#plt.axvline(normalized_cutoff_freq, color='green') # cutoff frequency
#plt.show()


avg_pert = []
avg_tx = []
avg_tx_pert = []
avg_tx_pert_wgn = []
avg_tx_pert_wgn_lpf = []


gaussian_prod = []
single_prod = []
bl_single_prod = []
fgsm_prod = []
bl_fgsm_prod = []

rxgaussian_prod = []
rxsingle_prod = []
rxbl_single_prod = []
rxfgsm_prod = []
rxbl_fgsm_prod = []

rx_grads = []
snr_list = []

# CIFAR-10 has 6000 samples/class. CIFAR-100 has 600. Somewhere in there seems like right order of magnitude
nvecs_per_key = 500
vec_length = 128
snr_vals = range(-20,20,2)
snr_vals = np.array([10.])
for D in t_offs:
 print("Stdev T is ",D)
 for cfo_lim in cfo_lims:
  print ("PPM is ", cfo_lim)
  for snr in snr_vals:
   for EpEs in EpEs_list:
    epsilon = np.sqrt(np.power(10.0, EpEs/10.0) / float(sps))
    print ("snr is ", snr)
    for alphabet_type in transmitters.keys():
        for i,mod_type in enumerate(transmitters[alphabet_type]):
          if mod_type.modname=='QPSK':
                  i = np.where(mod_type.modname==mods)
		  onehot = np.zeros(11)
		  onehot[i] = 1
		  print(onehot)
		  dataset[(mod_type.modname, EpEs)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)
		  control_d[(mod_type.modname, EpEs)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)
		  # moar vectors!
		  insufficient_modsnr_vectors = True
		  modvec_indx = 0
                  first_run = True
		  while insufficient_modsnr_vectors:
		      tx_len = int(10e3)
		      if mod_type.modname == "QAM16":
		          tx_len = int(20e3)
		      if mod_type.modname == "QAM64":
		          tx_len = int(30e3)

		      src = source_alphabet(alphabet_type, tx_len, True)
                      #ber_src = source_alphabet(alphabet_type, tx_len, True)
                      #ber_snk = blocks.vector_sink_b()
                      #ber_tb = gr.top_block()
                      #ber_tb.connect(ber_src, ber_snk)
                      #ber_tb.run()
                      #np.save('bits.npy', np.array(ber_snk.data(), dtype=np.float))

		      mod = mod_type()
		      snk = blocks.vector_sink_c()
		      tb = gr.top_block()
		      # connect blocks
		      tb.connect(src, mod, snk) # without channel
		      tb.run()

		      clean_output_vector = np.array(snk.data(), dtype=np.complex64)    

                      #temp = np.array(snk.data(), dtype=np.complex64) 
                      #temp1 = np.array(snk.data(), dtype=np.complex64)       
                      #single_fft_plot(clean_output_vector,'modulated','b','s')

                      # interpolate 
                      zeros = np.zeros(sps*len(clean_output_vector), dtype=np.complex64)
                      zeros[::sps] = clean_output_vector
                      clean_output_vector = zeros

                      #zeros = np.zeros(sps*len(temp), dtype=np.complex64)
                      #zeros[::sps] = temp
                      #temp = zeros

                      #zeros = np.zeros(sps*len(temp1), dtype=np.complex64)
                      ##zeros[::sps] = temp1
                      #temp1 = zeros
                      #fft_plot(clean_output_vector,'interpolated')

                      # PUlse shape   
                      clean_output_vector = np.convolve(clean_output_vector, rrc_taps)                
                      #temp = np.convolve(temp, rrc_taps)
                      #temp1 = np.convolve(temp1, rrc_taps)                                
                      #single_fft_plot(clean_output_vector,'Pulse shaped','b','s') 

                      # make sure Es=1 on average
                      clean_output_vector = np.sqrt(tx_len) * clean_output_vector / (np.linalg.norm(clean_output_vector)+sys.float_info.min)

                      energy = np.sum(np.abs(clean_output_vector)**2)
                      

                      #temp = np.sqrt(tx_len) * temp / (np.linalg.norm(temp)+sys.float_info.min)
                      #temp1 = np.sqrt(tx_len) * temp1 / (np.linalg.norm(temp1)+sys.float_info.min)
                      #fft_plot(clean_output_vector,'Energy normalized') 

                      # dither, sys.float_info.min
                      #clean_output_vector += np.random.normal(0,1e-40,clean_output_vector.shape)  + 1j*np.random.normal(0,1e-40,clean_output_vector.shape)
                      #avg_tx.append(temp)
                      #plt.plot(clean_output_vector, label='Tx')
                      #fft_plot(clean_output_vector,'Tx') 
     
		      # add FGSM to clean data, save to file
		      idx = 0 # start late to avoid transients
		      while idx + vec_length < len(clean_output_vector):
		          sampled_clean_vector = clean_output_vector[idx:idx+vec_length]
                          # power scale (acc test stage only)
                          scaled_vector = np.sqrt(2*len(sampled_clean_vector)) * sampled_clean_vector / np.linalg.norm(sampled_clean_vector)
		          reshape_x = np.concatenate((np.expand_dims(np.real(scaled_vector), 0),np.expand_dims(np.imag(scaled_vector), 0)), axis=0)
                          #print(np.sum(np.abs(reshape_x)**2))
                          reshape_x = np.expand_dims(np.expand_dims(reshape_x, axis=-1), axis=0)
		          x_tensor = tf.convert_to_tensor(reshape_x, dtype=tf.float32)
		          with tf.GradientTape() as t:
		              t.watch(x_tensor)
		              output = model(x_tensor)
		              loss = tf.keras.losses.categorical_crossentropy(tf.convert_to_tensor(np.expand_dims(onehot,axis=0), dtype=tf.float32), output)

		          gradients = t.gradient(loss, x_tensor)

                          if attack=='fgsm':  # l_inf=epsilon norm attack
                              pert = tf.sign(gradients)
                              scaled_pert = epsilon*pert
                              # convert back to a (128,) compl
                              # only keep elements that match signal's sign
                              real = scaled_pert[0,0,:,0].numpy()
                              #real[np.array(np.sign(reshape_x[0,0,:,0]) != np.sign(gradients[0,0,:,0]))] = 0
                              imag = scaled_pert[0,1,:,0].numpy()
                              #imag[np.array(np.sign(reshape_x[0,1,:,0]) != np.sign(gradients[0,1,:,0]))] = 0
                              scaled_pert = real + 1j * imag
                              Ps = np.sum(np.abs(sampled_clean_vector)**2)
                              Pp = Ps * pow(10.0, EpEs / 10.0)
                              scaled_pert= np.sqrt(Pp) * (scaled_pert / np.linalg.norm(scaled_pert))
                              fgsm_pert = scaled_pert 


                          if attack=='bl-fgsm':
                              pert = tf.sign(gradients)
                              scaled_pert = epsilon*pert
                              # convert back to a (128,) compl
                              real = scaled_pert[0,0,:,0].numpy()
                              #real[np.array(np.sign(reshape_x[0,0,:,0]) != np.sign(gradients[0,0,:,0]))] = 0
                              imag = scaled_pert[0,1,:,0].numpy()
                              #imag[np.array(np.sign(reshape_x[0,1,:,0]) != np.sign(gradients[0,1,:,0]))] = 0
                              scaled_pert = real + 1j * imag

                              scaled_pert = np.convolve(scaled_pert, rrc_taps, 'same')
                              Ps = np.sum(np.abs(sampled_clean_vector)**2)
                              Pp = Ps * pow(10.0, EpEs / 10.0)
                              scaled_pert= np.sqrt(Pp) * (scaled_pert / np.linalg.norm(scaled_pert))
                              bl_fgsm_pert = scaled_pert 

                          if attack=='awgn':  # control perturbation
         		      pert_amp = pow(pow(10.0, EpEs / 10.0) / (sps), 0.5)
                              scaled_pert = np.random.normal(0,pert_amp,len(sampled_clean_vector)) + 1j*np.random.normal(0,pert_amp,len(sampled_clean_vector))

                              awgn_pert = scaled_pert  

                          if attack=='boost':  # control perturbation
                              pert = sampled_clean_vector
                              Ps = np.sum(np.abs(sampled_clean_vector)**2)
                              Pp = Ps * pow(10.0, EpEs / 10.0)
                              scaled_pert = np.sqrt(Pp) * (pert / np.linalg.norm(pert))
                              awgn_pert = scaled_pert

                          if attack=='single':  # l_0=1 norm attack
                              array_grad_real = gradients[0,0,:,0].numpy()
                              center_real = np.argmax(np.abs(np.array(array_grad_real[:])))

                              #while np.sign(array_grad_real[center_real]) != np.sign(reshape_x[0,0,center_real,0]):
                                  #array_grad_real[center_real] = 0.
                                  #center_real = np.argmax(np.abs(np.array(array_grad_real[:])))
                              
                              array_grad_imag = gradients[0,1,:,0].numpy()
                              center_imag = np.argmax(np.abs(np.array(array_grad_imag[:])))
                              #while np.sign(array_grad_imag[center_imag]) != np.sign(reshape_x[0,1,center_imag,0]):
                                  #array_grad_imag[center_imag] = 0.
                                  #center_imag = np.argmax(np.abs(np.array(array_grad_imag[:])))

                              pert = signal.unit_impulse(len(sampled_clean_vector), center_real)*np.sign(np.array(gradients[0,0,center_real,0])) + np.sign(np.array(gradients[0,1,center_imag,0]))*signal.unit_impulse(len(sampled_clean_vector), center_imag)*1j
                              scaled_pert=np.sqrt(len(sampled_clean_vector))*epsilon*pert
                              single_pert = scaled_pert
                              

                          if attack=='bl-single':  # modified l_0=N norm attack
                              array_grad = gradients.numpy()
                              center_real = np.argmax(np.abs(np.array(array_grad[0,0,:,0])))
                              #while np.sign(array_grad[0,0,center_real,0]) != np.sign(reshape_x[0,0,center_real,0]):
                                  #array_grad[0,0,center_real,0] = 0.
                                  #center_real = np.argmax(np.abs(np.array(array_grad[0,0,:,0])))
                              
                              center_imag = np.argmax(np.abs(np.array(array_grad[0,1,:,0])))
                              #while np.sign(array_grad[0,1,center_imag,0]) != np.sign(reshape_x[0,1,center_imag,0]):
                                  #array_grad[0,1,center_imag,0] = 0.
                                  #center_imag = np.argmax(np.abs(np.array(array_grad[0,1,:,0])))

                              pert = signal.unit_impulse(len(sampled_clean_vector), center_real)*np.sign(np.array(gradients[0,0,center_real,0])) + np.sign(np.array(gradients[0,1,center_imag,0]))*signal.unit_impulse(len(sampled_clean_vector), center_imag)*1j

                              # BL it with RRC
                              pert = np.convolve(pert, rrc_taps, 'same')
                              # little harder to scale
                              Ps = np.sum(np.abs(sampled_clean_vector)**2)
                              Pp = Ps * pow(10.0, EpEs / 10.0)
                              scaled_pert= np.sqrt(Pp) * (pert / np.linalg.norm(pert))
                              
                              bl_single_pert = scaled_pert

                          if attack=='2-bl-single':
                              center_real = list(np.sort(np.abs(np.array(gradients[0,0,:,0])).argsort()[-p:][::-1]))
                              center_imag = list(np.sort(np.abs(np.array(gradients[0,1,:,0])).argsort()[-p:][::-1]))

                              imp_train_re = np.zeros(len(sampled_clean_vector))
                              imp_train_re[center_real] = 1.
                              imp_train_im = np.zeros(len(sampled_clean_vector))
                              imp_train_im[center_imag] = 1.
                              list_grad = np.array(gradients)
                              sgn_real = np.zeros(len(sampled_clean_vector))
                              sgn_real[center_real] = np.sign(list_grad[0,0,center_real,0])
                              sgn_imag = np.zeros(len(sampled_clean_vector))
                              sgn_imag[center_imag] = np.sign(list_grad[0,1,center_real,0])
                              pert = imp_train_re*sgn_real + sgn_imag*imp_train_im*1j

                              # BL it with RRC
                              pert = np.convolve(pert, rrc_taps, 'same')
                              # little harder to scale
                              Ps = np.sum(np.abs(sampled_clean_vector)**2)
                              Pp = Ps * pow(10.0, EpEs / 10.0)
                              scaled_pert= np.sqrt(Pp) * (pert / np.linalg.norm(pert))
                              
                              p_bl_single_pert = scaled_pert
        


                          
		          #print(10*np.log10(np.sum(np.abs(temp[idx:idx+vec_length])**2)/np.sum(np.abs(scaled_pert)**2))) 
                          #np.save('sinc_pert',scaled_pert)


                          #fig, ax1 = plt.subplots()
                          #color = 'tab:blue'
                          #ax1.set_xlabel('Sample [n]')
                          #ax1.set_ylabel('Gradient',color=color)
                          #ax1.plot(np.sign(np.array(gradients[0,1,:,0]) / np.sum(np.abs(np.array(gradients[0,1,:,0])))), color=color)
                          

                          #ax2 = ax1.twinx()
                          #color = 'tab:orange'
                          #ax2.set_ylabel('Perturbation',color=color)
                          #ax1.plot(np.real(awgn_pert), color='y', label='Boost', linestyle='--')
                          #ax1.plot(np.real(fgsm_pert), color='tab:orange', label='FGSM', linestyle='-.')
                          #ax1.plot(np.real(bl_fgsm_pert), color='b', label='BL FGSM', linestyle='-')
                          #ax1.plot(np.real(single_pert), color='r', label='Single', linestyle=':')
                          #ax1.plot(np.real(bl_single_pert), color='m', label='BL Single', linestyle='-')
                          #ax1.plot(np.real(sampled_clean_vector),label='Signal',color='black',linewidth=2)
                          #ax1.tick_params(axis='y', labelcolor=color)
                          
                          #ax2.tick_params(axis='y', labelcolor=color)
                          #plt.grid()
                          #plt.legend()
                          #plt.show()

                          
                          #single_fft_plot(awgn_pert,'Gaussian','y','X')
                          #single_fft_plot(fgsm_pert,'FGSM','tab:orange','^')
                          #single_fft_plot(bl_fgsm_pert,'BL FGSM','b','s')
                          #single_fft_plot(single_pert,'Single Sample','r','*')
                          #single_fft_plot(bl_single_pert,'BL Single','m','s')
                          #single_fft_plot(sampled_clean_vector,'Signal','k','o')


                          #plt.legend()
                          #plt.grid()
                          #plt.show()

                          # dot products
                          #complex_grad = np.array(gradients[0,0,:,0])+1j*np.array(gradients[0,1,:,0])
                          #gaussian_prod.append(np.dot(np.real(awgn_pert),np.real(complex_grad))+1j*np.dot(np.imag(awgn_pert),np.imag(complex_grad)))
                          #single_prod.append(np.dot(np.real(single_pert),np.real(complex_grad))+1j*np.dot(np.imag(single_pert),np.imag(complex_grad)))
                          #bl_single_prod.append(np.dot(np.real(bl_single_pert),np.real(complex_grad))+1j*np.dot(np.imag(bl_single_pert),np.imag(complex_grad)))
                          #fgsm_prod.append(np.dot(np.real(fgsm_pert),np.real(complex_grad))+1j*np.dot(np.imag(fgsm_pert),np.imag(complex_grad)))

                          #bl_fgsm_prod.append(np.dot(np.real(bl_fgsm_pert),np.real(complex_grad))+1j*np.dot(np.imag(bl_fgsm_pert),np.imag(complex_grad)))


                          # channel model for violin plots
                          noise_amp = pow(pow(10.0, -snr / 10.0), 0.5)                      

                          dot_noise = np.random.normal(0,noise_amp,len(scaled_pert)) + 1j*np.random.normal(0,noise_amp,len(scaled_pert))
                          samples = truncnorm.rvs(-cfo_lim*fs/(1e6*len(scaled_pert)), cfo_lim*fs/(1e6*len(scaled_pert)), scale=100., size=len(scaled_pert))
                          cfo = np.cumsum(samples)
                          time_range = np.array(range(0,len(scaled_pert)/2) + range(-len(scaled_pert)/2+1,0+1),dtype=np.complex128)
                          samples = truncnorm.rvs(-D, D, scale=100., size=len(scaled_pert))
                          wtao = np.cumsum(samples)*2*np.pi/len(scaled_pert)*time_range
                          shift = np.cos(wtao) - 1j*np.sin(wtao)

                          # compute RX gradient
                          sampled_rx = np.fft.ifft(np.fft.fft(dot_noise + sampled_clean_vector+scaled_pert)*shift) * np.power(np.e, -1j*2*np.pi*cfo/fs*range(len(sampled_clean_vector)))
                          scaled_rx = np.sqrt(2*len(sampled_rx)) * sampled_rx / np.linalg.norm(sampled_rx)

                          reshape_x = np.expand_dims(np.expand_dims(np.concatenate((np.expand_dims(np.real(scaled_rx), 0),np.expand_dims(np.imag(scaled_rx), 0)), axis=0),-1),0)
		          x_tensor = tf.convert_to_tensor(reshape_x, dtype=tf.float32)
		          with tf.GradientTape() as t:
		              t.watch(x_tensor)
		              output = model(x_tensor)
		              loss = tf.keras.losses.categorical_crossentropy(tf.convert_to_tensor(np.expand_dims(onehot,axis=0), dtype=tf.float32), output)

		          rx_gradients = t.gradient(loss, x_tensor)
                          rx_complex_grad = np.array(rx_gradients[0,0,:,0])+1j*np.array(rx_gradients[0,1,:,0])
                          rx_grads.append(rx_complex_grad.flatten())
                          snr_list.append(np.repeat(EpEs,len(rx_complex_grad)))


                          # received signal dot products
                          #temp = np.fft.ifft(np.fft.fft(dot_noise+awgn_pert* np.power(np.e, -1j*2*np.pi*cfo/fs*range(len(awgn_pert))))*shift)
                          #rxgaussian_prod.append(np.dot(np.real(temp),np.real(rx_complex_grad))+1j*np.dot(np.imag(temp),np.imag(rx_complex_grad)))

                          #temp = np.fft.ifft(np.fft.fft(dot_noise+single_pert* np.power(np.e, -1j*2*np.pi*cfo/fs*range(len(single_pert))))*shift)
                          #rxsingle_prod.append(np.dot(np.real(temp),np.real(rx_complex_grad))+1j*np.dot(np.imag(temp),np.imag(rx_complex_grad)))
                          #temp = np.fft.ifft(np.fft.fft(dot_noise+bl_single_pert* np.power(np.e, -1j*2*np.pi*cfo/fs*range(len(bl_single_pert))))*shift)
                          #rxbl_single_prod.append(np.dot(np.real(temp),np.real(rx_complex_grad))+1j*np.dot(np.imag(temp),np.imag(rx_complex_grad)))

                          #temp = np.fft.ifft(np.fft.fft(dot_noise+fgsm_pert* np.power(np.e, -1j*2*np.pi*cfo/fs*range(len(fgsm_pert))))*shift)
                          #rxfgsm_prod.append(np.dot(np.real(temp),np.real(rx_complex_grad))+1j*np.dot(np.imag(temp),np.imag(rx_complex_grad)))
                          #temp = np.fft.ifft(np.fft.fft(dot_noise+bl_fgsm_pert* np.power(np.e, -1j*2*np.pi*cfo/fs*range(len(bl_fgsm_pert))))*shift)
                          #rxbl_fgsm_prod.append(np.dot(np.real(temp),np.real(rx_complex_grad))+1j*np.dot(np.imag(temp),np.imag(rx_complex_grad)))

                          #temp1[idx:idx+vec_length] += scaled_pert
                          
                          Ps = np.sum(np.abs(clean_output_vector[idx:idx+vec_length])**2)
                          ratio  = 10*np.log10(Ps / np.sum(np.abs(scaled_pert)**2))
                          clean_output_vector[idx:idx+vec_length] += scaled_pert
                          # don't add any energy with perturbation
                          #clean_output_vector[idx:idx+vec_length] = np.sqrt(Ps) * (clean_output_vector[idx:idx+vec_length] / np.linalg.norm(clean_output_vector[idx:idx+vec_length]))
                          
                          

		          idx += vec_length # increase counter

                      print('Es/Ep: '+str(ratio))

                      #single_fft_plot(temp1,'FGSM','b','s')
                      #fft_plot(clean_output_vector,'Tx+FGSM') 
                      #avg_tx_pert.append(temp1)   
                      #plt.plot(clean_output_vector, label='pert')

		      # save to file to avoid writing an OOT
		      #np.save('temp',clean_output_vector)
		      #file_src = blocks.file_source(len(clean_output_vector),'temp.npy',False)
		      #v_2_s = blocks.vector_to_stream(sps,int(len(clean_output_vector)/sps))
		      #snk3 = blocks.vector_sink_c()
		      
		      noise_amp = pow(pow(10.0, -snr / 10.0), 0.5)                  

		      #tb3 = gr.top_block()
		      #tb3.connect(file_src, v_2_s, snk3)
                      #src2 = source_alphabet(alphabet_type, tx_len, True)
		      #mod2 = mod_type()
                      #tb3.connect(src2, mod2, chan, rrc_filter, snk3)
		      #tb3.run()

		      raw_output_vector = clean_output_vector #np.array(snk3.data(), dtype=np.complex64)

                      noise = np.random.normal(0,noise_amp,len(raw_output_vector)) + 1j*np.random.normal(0,noise_amp,len(raw_output_vector))
                      Ps = np.sum(np.abs(raw_output_vector)**2)
                      Pn = np.sum(np.abs(noise)**2)

                      print('SNR: '+str(10*np.log10(sps*energy/Pn)))
                      print('S+P NR: '+str(10*np.log10(sps*Ps/Pn)))
                      # AWGN
                      #plt.plot(raw_output_vector, label='Rx')
                      raw_output_vector += noise
                      #single_fft_plot(raw_output_vector,'Tx+Pert+AWGN','b','s') 
                      #avg_tx_pert_wgn.append(raw_output_vector)    
                      #plt.plot(raw_output_vector, label='Rx')

                      # time shift by half samples
                      time_range = np.array(range(0,len(raw_output_vector)/2) + range(-len(raw_output_vector)/2+1,0+1),dtype=np.complex128)
                      samples = truncnorm.rvs(-D, D, scale=100., size=len(raw_output_vector))
                      wtao = np.cumsum(samples)*2*np.pi/len(raw_output_vector)*time_range
                      shift = np.cos(wtao) - 1j*np.sin(wtao)
                      raw_output_vector = np.fft.ifft(np.fft.fft(raw_output_vector)*shift)


                      # CFO
                      samples = truncnorm.rvs(-cfo_lim*fs/(1e6*128), cfo_lim*fs/(1e6*128), scale=100., size=len(raw_output_vector))
                      cfo = np.cumsum(samples)
                      #plt.plot(cfo)
                      #plt.xlabel('Sample [n]')
                      #plt.ylabel('Offset (Hz)')
                      #plt.grid()
                      #plt.show()
                      
                      raw_output_vector = raw_output_vector * np.power(np.e, -1j*2*np.pi*cfo/fs*range(len(raw_output_vector)))
                      
                      # MF
                      #raw_output_vector = np.convolve(raw_output_vector, rrc_taps)
                      #single_fft_plot(raw_output_vector,'MatchedFiltered','b','s')
                      #plt.plot(raw_output_vector)
                      #plt.show()

                      # LPF
                      raw_output_vector = scipy.signal.filtfilt(numerator_coeffs, denominator_coeffs, raw_output_vector)
                      #single_fft_plot(raw_output_vector, 'LPF','b','s')
                      #avg_tx_pert_wgn_lpf.append(raw_output_vector)
                      #plt.plot(raw_output_vector)
                      #plt.legend(['Rx','LPF'])
                      #plt.legend()
                      #plt.grid()
                      #plt.show()
                      #single_fft_plot(raw_output_vector,'Tx+Pert+AWGN+LPF','k','^')
                      
                      #plt.show()

                      #np.save('received'+str(attack)+str(EpEs)+str(snr)+'.npy', raw_output_vector)  

		      # start the sampler some random time after channel model transients (arbitrary values here)
		      sampler_indx = random.randint(50, 500)
		      while sampler_indx + vec_length < len(raw_output_vector) and modvec_indx < nvecs_per_key:
		          sampled_vector = raw_output_vector[sampler_indx:sampler_indx+vec_length]
                          # power norm (acc test stage only)
                          sampled_vector = np.sqrt(2*len(sampled_vector)) * sampled_vector / np.linalg.norm(sampled_vector)
                          
		          dataset[(mod_type.modname, EpEs)][modvec_indx,0,:] = np.real(sampled_vector)
		          dataset[(mod_type.modname, EpEs)][modvec_indx,1,:] = np.imag(sampled_vector)
		          
		          # bound the upper end very high so it's likely we get multiple passes through
		          # independent channels
		          sampler_indx += random.randint(vec_length, round(len(raw_output_vector)*.05))
		          modvec_indx += 1
                          #print(modvec_indx)
                      print('Signal Energy: '+str(np.sum(np.abs(sampled_vector)**2)))

		      if modvec_indx == nvecs_per_key:
		          # we're all done
		          insufficient_modsnr_vectors = False
                          

print ("all done. writing to disk")
cPickle.dump( dataset, file("RML2021.10a.varyP."+str(attack)+"_"+str(EpEs)+".dat", "wb" ) )
#cPickle.dump( dataset, file("RML2021.10a_dict.dat", "wb" ) )


import seaborn as sns
import pandas as pd
#flattened_data = np.array([np.real(single_prod)+np.imag(single_prod),np.real(rxsingle_prod)+np.imag(rxsingle_prod), np.real(bl_single_prod)+np.imag(bl_single_prod),np.real(rxbl_single_prod)+np.imag(rxbl_single_prod), np.real(gaussian_prod)+np.imag(gaussian_prod),np.real(rxgaussian_prod)+np.imag(rxgaussian_prod),np.real(fgsm_prod)+np.imag(fgsm_prod),np.real(rxfgsm_prod)+np.imag(rxfgsm_prod),np.real(bl_fgsm_prod)+np.imag(bl_fgsm_prod),np.real(rxbl_fgsm_prod)+np.imag(rxbl_fgsm_prod)]).flatten()


#types = np.repeat(['Tx','AWGN','Tx','AWGN','Tx','AWGN','Tx','AWGN','Tx','AWGN'],len(single_prod))
#names = np.repeat(['Single','Single','BL Single','BL Single','None','None','FGSM','FGSM','BL FGSM','BL FGSM'],len(single_prod))

#data_frame = pd.DataFrame({'Sum of Complex Dot Product':flattened_data, 'Signal':types, 'Perturbation':names})

#sns.violinplot(data=data_frame,x='Perturbation',y='Sum of Complex Dot Product',hue='Signal',split=True,palette='muted',scale='width',linewidth=0)
#plt.grid()
#plt.tight_layout()
#plt.show()

snr_list = np.array(snr_list).flatten()
rx_grads = np.array(rx_grads).flatten()

data_frame = pd.DataFrame({'SNR':snr_list,'gradients':rx_grads})
np.save('awgn_grads',rx_grads)
np.save('grads_snr',snr_list)
# Plot the responses for different events and regions
sns.lineplot(x="SNR", y="gradients",data=data_frame)
plt.grid()
plt.show()

#avg_pert = np.array(avg_pert)
#avg_tx = np.array(avg_tx)
#avg_tx_pert = np.array(avg_tx_pert)
#avg_tx_pert_wgn = np.array(avg_tx_pert_wgn)
#avg_tx_pert_wgn_lpf = np.array(avg_tx_pert_wgn_lpf)


#fft_plot(avg_tx,'Tx')
#fft_plot(avg_pert,'Pert') 
#fft_plot(avg_tx_pert,'Tx+Pert') 
#plt.legend()
#plt.grid()
#plt.show()
#fft_plot(avg_tx_pert_wgn,'Tx+Pert+AWGN') 
#fft_plot(avg_tx_pert_wgn_lpf,'Tx+Pert+AWGN+LPF') 
#plt.legend()
#plt.grid()
#plt.show()









