#!/usr/bin/env python
from transmitters import transmitters
from source_alphabet import source_alphabet
import analyze_stats
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, pickle, gzip
import random
import keras


'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True

dataset = {}

model = keras.models.load_model('VT-CNN2')  # trained model

I = 8  # interpolation factor of RML2016.10a (sps)
Ep = 0  # target Ep/Es ratio (dB)
epsilon = np.sqrt(np.power(10, Ep/10) / (2*I))  # magnitude of each sample in the pert

# The output format looks like this
# {('mod type', SNR): np.array(nvecs_per_key, 2, vec_length), etc}

# CIFAR-10 has 6000 samples/class. CIFAR-100 has 600. Somewhere in there seems like right order of magnitude
nvecs_per_key = 1000
vec_length = 128
snr_vals = range(-20, 20, 2)
for snr in snr_vals:
    print("snr is ", snr)
    for alphabet_type in transmitters.keys():
        for i,mod_type in enumerate(transmitters[alphabet_type]):
          dataset[(mod_type.modname, snr)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)
          # moar vectors!
          insufficient_modsnr_vectors = True
          modvec_indx = 0
          while insufficient_modsnr_vectors:
              tx_len = int(10e3)
              if mod_type.modname == "QAM16":
                  tx_len = int(20e3)
              if mod_type.modname == "QAM64":
                  tx_len = int(30e3)
              src = source_alphabet(alphabet_type, tx_len, True)
              mod = mod_type()
              fD = 1
              delays = [0.0, 0.9, 1.7]
              mags = [1, 0.8, 0.3]
              ntaps = 8
              noise_amp = 10**(-snr/10.0)
              chan = channels.dynamic_channel_model(samp_rate=200e3, sro_std_dev=0.01, sro_max_dev=50, cfo_std_dev=.01,
                                                    cfo_max_dev=0.5e3, N=8, doppler_freq=fD, LOS_model=True, K=4,
                                                    delays=delays, mags=mags, ntaps_mpath=ntaps, noise_amp=noise_amp,
                                                    noise_seed=0x1337)

              snk = blocks.vector_sink_c()
              snk2 = blocks.vector_sink_c()

              tb = gr.top_block()
              tb2 = gr.top_block()

              # connect blocks
              if apply_channel:
                  tb.connect(src, mod, chan, snk)
                  tb2.connect(src, mod, snk2)
              else:
                  tb.connect(src, mod, snk)
              tb.run()
              tb2.run()

              clean_output_vector = np.array(snk2.data(), dtype=np.complex64)
              raw_output_vector = np.array(snk.data(), dtype=np.complex64)

              # start the sampler some random time after channel model transients (arbitrary values here)
              sampler_indx = random.randint(50, 500)
              while sampler_indx + vec_length < len(raw_output_vector) and modvec_indx < nvecs_per_key:
                  sampled_vector = raw_output_vector[sampler_indx:sampler_indx+vec_length]
                  sampled_clean_vector = clean_output_vector[sampler_indx:sampler_indx + vec_length]

                  # Normalize the energy in this vector to be 1
                  energy = np.sum((np.abs(sampled_vector)))
                  energy2 = np.sum((np.abs(sampled_clean_vector)))
                  sampled_vector = sampled_vector / energy

                  # clean vector with which to compute the FGSM perturbation
                  sampled_clean_vector = sampled_clean_vector / energy2
                  x_tensor = tf.convert_to_tensor(sampled_clean_vector, dtype=tf.float32)
                  with tf.GradientTape() as t:
                      t.watch(x_tensor)
                      output = model(x_tensor)
                      loss = keras.losses.categorical_crossentropy(mod_type.modname, output[0])
                  gradients = t.gradient(loss, x_tensor)
                  pert = tf.sign(gradients)
                  scaled_pert = epsilon * pert
                  sampled_vector += scaled_pert

                  # add vector to dataset
                  dataset[(mod_type.modname, snr)][modvec_indx,0,:] = np.real(sampled_vector)
                  dataset[(mod_type.modname, snr)][modvec_indx,1,:] = np.imag(sampled_vector)

                  # bound the upper end very high so it's likely we get multiple passes through
                  # independent channels
                  sampler_indx += random.randint(vec_length, round(len(raw_output_vector)*.05))
                  modvec_indx += 1

              if modvec_indx == nvecs_per_key:
                  # we're all done
                  insufficient_modsnr_vectors = False

print("all done. writing to disk")
pickle.dump(dataset, file("RML2016.10a_dict.dat", "wb"))
