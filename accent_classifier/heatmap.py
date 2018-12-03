from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os
import pickle
import keras
import matplotlib.pyplot as plt

from vis.visualization import visualize_activation

from vis.visualization import visualize_cam
from vis.utils import utils
from keras import activations

import scipy.io.wavfile as wavfile
from scipy.io.wavfile import read
from scipy import signal
import glob
import copy
import sys
'''
opening comment:

we load a trained model in .h5 format, and pass an example image through the network
to find the hidden layer weights visualizations


The main function takes 1sec_hd pickles and loop through 20 of them,
concatenate all the images

each layer can be visualized, or choose to visualize a particular layer
'''


def concatIm(imArr):
    '''
    input: n lists of list of numpy arrays of 202 by 66 by 3 or not 3
    output: 202 by 66n by 3
    '''
    #res = np.ndarray(shape=(imArr[0][0].shape[1], len(imArr) *  imArr[0][0].shape[2], 3))


    res = np.concatenate((imArr), axis = 1)
    if len(res.shape) == 3:
        res = res.reshape(res.shape[1], res.shape[0], res.shape[2])
    else:
        res = res.reshape(res.shape[1], res.shape[0])
    print("res shape in concatIm", res.shape)
    return res

def plotSave(imFinal,figName, output_dir):
    '''
    after aggregating all 1 sec segments
    save it into a bigger plot
    '''

    plt.figure()
    print("trying to plot original spec ", imFinal.shape)
    if imFinal.shape[2] == 3:
        prev_shape = imFinal.shape
        imFinal -= np.min(imFinal)
        imFinal = np.minimum(imFinal, 255)
        imFinal = imFinal.reshape(prev_shape[1],prev_shape[0], 3)
        plt.imshow(imFinal, origin='lower')
    else:
        prev_shape = imFinal.shape
        imFinal = imFinal.reshape(prev_shape[1],prev_shape[0])
        # try to get the original plot to have more obvious colours
        imFinal -= np.min(imFinal)
        imFinal = np.minimum(imFinal, 255)

        plt.imshow(imFinal, cmap = plt.get_cmap('viridis'))


    #plt.xticks(np.arange(0, imFinal.shape[0], step=100))
    plt.xticks(np.arange(0, imFinal.shape[0], step = 66), np.arange(0, 20))

    plt.xlabel('Time (ms)')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '{}.png'.format(figName)))
    #plt.show()


def wav_2_spec(file):
    '''
    taken from Jonathan's preprocessing
    '''
    os.chdir("./cnn_vis")
    # will only plot the 1_0.wav in the dir
    file_name = "1_0_20sec"
    print("wav to spec", file_name)
    x_value = 0
    sr_value = 0
    sr_value, x_value = read(file_name+".wav")
    img = 0

    spectrum, specs, t, img = plt.specgram(x_value, NFFT=400, Fs=16000,  noverlap=160)


    plt.savefig("saved_spectrogram_20_sec_wav.png")
    plt.clf()
    #time.sleep(2)

    plt.close()
    spect_dict = {}
    spect_dict['img'] = img
    spect_dict['t'] = t
    spect_dict['specs'] = specs
    spect_dict['spectrum'] = spectrum
    print("spectrum", spectrum.shape)
    spectrum -= np.min(spectrum)
    spectrum = np.minimum(spectrum, 255)


    # plt.imshow(spectrum, cmap = plt.get_cmap('viridis'), origin='lower')
    # plt.tight_layout()
    # plt.xticks(np.arange(0, spectrum.shape[0], step = 66), np.arange(0, 20))
    #
    # plt.xlabel('Time (ms)')
    # plt.yticks([])
    # plt.savefig("saved_spectrum_1_0_20_sec_wav_cmap_normalized_flipped.png")
    #
    # #plt.show()
    # #time.sleep(2)
    #
    # plt.clf()
    # plt.close()
    # back to the prevs
    os.chdir("../")
    return spectrum



if __name__ == '__main__':
    print("sys.argv", sys.argv)
    if sys.argv[1] == 'plot':
        print("in plot")
        with open("heatmap_primitive_arr.pickle", 'rb') as jar:
            heatmap = pickle.load(jar)
    else:

        DATA_PATH = "./cnn_vis/1_0_20sec.wav"

        input_dir = "./cnn_vis/1sec_hd_pickle"

        os_list = os.listdir(input_dir)
        final_test_data = []
        for i, filename in enumerate(os_list):
            with open(os.path.join(input_dir, filename), 'rb') as jar:
                spect_dict = (pickle.load(jar))
                data = spect_dict['spectrum']
            final_test_data.append(data)
        final_test_data = np.array(final_test_data)
        print("final_test_data", final_test_data.shape)

        MODEL_PATH = './'

        model = keras.models.load_model(os.path.join(MODEL_PATH , 'val_acc_87_30_epoches.h5'))
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        print("The model layers dictionary: \n", layer_dict)


        final_test_data = np.expand_dims(final_test_data, axis=3)
        print("final_test_data", final_test_data.shape)

        layers_arr = ['conv2d_38', 'conv2d_39', 'dropout_23'] #just plot the last convolution layer
        for layerName in layers_arr:
            aggreCam = []

            for i in range(20):
                # Utility to search for layer index by name.
                # Alternatively we can specify this as -1 since it corresponds to the last layer.
                layer_idx = utils.find_layer_idx(model, layerName)

                # Swap softmax with linear
                model.layers[layer_idx].activation = activations.linear
                model = utils.apply_modifications(model)

                # This is the output node we want to maximize.
                filter_idx = 0

                img = visualize_cam(model, layer_idx, filter_indices=[0],
                 seed_input = final_test_data[i], penultimate_layer_idx = layer_idx - 1)


                aggreCam.append(img)

            heatmap = concatIm(aggreCam)

            # use the same 1_0.wav as the x_val to compare across CNN, RNN

            spectrum = wav_2_spec(DATA_PATH)
            print("spectrum", spectrum.shape)


            plt.imshow(spectrum, origin='lower', cmap=plt.cm.gray)

            print("heatmap shape", heatmap.shape)
            heatmap = heatmap.reshape(heatmap.shape[1], heatmap.shape[0], heatmap.shape[2])
            plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=.6,  origin='lower')
            plt.savefig("overlayed_{}_{}sec.png".format(layerName, i))
            #plt.show()


            # works for continuous plot but has issue with x axis shrinking
            with open("heatmap_primitive_arr.pickle", 'wb') as jar:
                pickle.dump(heatmap, jar, protocol=pickle.HIGHEST_PROTOCOL)

            plotSave(heatmap, "cnn_filter_{}_{}sec_normalized_flipped".format(layerName, i), "./")
