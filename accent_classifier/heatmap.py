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

'''
opening comment:
Visualization of the gradients in CNN
adapted from https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py

we load a trained model in .h5 format, and pass an example image through the network
to find the hidden layer weights visualizations

The original implementation can ONLY work with keras pretrained VGG16

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
    # if len(res.shape) == 3:
    #     res = res.reshape(res.shape[1], res.shape[0], res.shape[2])
    # else:
    #     res = res.reshape(res.shape[1], res.shape[0])
    print("res shape in concatIm", res.shape)
    return res


def plotSave(imFinal,figName, output_dir):
    '''
    after aggregating all 1 sec segments
    save it into a bigger plot
    '''

    plt.figure()
    if len(imFinal.shape) == 3:
        plt.imshow( cv2.cvtColor(imFinal, cv2.COLOR_BGR2RGB), aspect="auto")
    else:

        plt.imshow(imFinal, cmap = 'gray', aspect="auto")


    plt.imshow(imFinal, aspect="auto")
    plt.xticks(np.arange(0, imFinal.shape[0], step=100))
    plt.xlabel('Time (ms)')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '{}.png'.format(figName)))
    #plt.show()




if __name__ == '__main__':

    DATA_PATH = "./"

    with open(os.path.join(DATA_PATH, "val_1_sec_hd_feature.pickle"), 'rb') as jar:
        x_val = pickle.load(jar)

    with open(os.path.join(DATA_PATH, "val_1_sec_hd_label.pickle"), 'rb') as jar:
        y_val = pickle.load(jar)

    print("Finished loading pickle")
    print(x_val.shape)
    print(y_val.shape)


    MODEL_PATH = './'

    aggreCam = []
    originalSpec = [] # plot the original spectrogram for reference
    model = keras.models.load_model(os.path.join(MODEL_PATH , 'val_acc_87_30_epoches.h5'))
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    print("The model layers dictionary: \n", layer_dict)

    foo = 1
    bar = 1
    while foo: # will replace with iterating through brit and libris classes
        foo = 0
        while bar:
            bar = 0
        # for layerName, layerVal in layer_dict.items():
            layerName = 'conv2d_39'
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
                 seed_input = x_val[i], penultimate_layer_idx = layer_idx - 1)

                #print("vis cam img", img.shape)

                # img shape = (201, 66, 3)
                # plt.imshow(img, cmap='jet')
                # #plt.imshow(img[..., 0])
                # figName = "multiclass_test2"
                # plt.savefig('kerasvis_{}.png'.format(figName))
                # plt.show()
                aggreCam.append(img)
            res = concatIm(aggreCam)
            # works for a single plot
            plotSave(res, "testpltLib_20sec_no_reshape", "./")
