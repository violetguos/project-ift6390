from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import sys
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
# Vi Comment: if i don't call this line, matplotlib crashes and gives me a call
# stack trace that triggers *literally and figuratively* me and my computer
from matplotlib import cm
import matplotlib.pyplot as plt

import torch

def plot_mfcc(mfcc_feat, title):
    '''
    plots one single mfcc feat file
    totally optional, we will see if we need it for CNN
    '''

    fig, ax = plt.subplots()

    # shape transposed, not sure if flipped data
    mfcc_data= np.swapaxes(mfcc_feat, 0 ,1)

    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
    ax.set_title('MFCC')
    #Showing mfcc_data
    plt.show()
    fig.savefig(title + '_data' + '.png')
    print("Saving MFCC plot for {}\n".format(title))

    # this only shows mfcc_feat but does not save
    # Showing mfcc_feat
    # plt.plot(mfcc_feat)
    # plt.show()



def get_mfcc():
    '''
    single file
    '''


    # change directory from code dir to data dir
    #
    os.chdir(sys.argv[1])
    # read only .wav files

    print("os.path.join(sys.argv[1], '*.wav')", os.path.join(sys.argv[1], '*.wav'))
    allWaves = glob.glob(os.path.join(sys.argv[1], '*.wav'))

    # hardcoded number for testing
    for i in range(1):#(len(allWaves)):
        (rate,sig) = wav.read(allWaves[i]) #(sys.argv[1] + "121_0.wav")
        mfcc_feat = mfcc(sig,rate, nfft=551)
        #print(mfcc_feat.shape)
        if i%100 == 0:
            print("Saving MFCC text for {}\n".format(allWaves[i]))
        # to get file name without '.wav' extension
        # works for any file if you want no '.' (os.path.splitext(allWaves[i])[0])
        np.savetxt(sys.argv[2] + os.path.splitext(allWaves[i])[0] + '.txt', mfcc_feat, delimiter=",")



        # only if we want to call plot function
        # plot_mfcc(mfcc_feat, sys.argv[3] + os.path.splitext(allWaves[i])[0] )









if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate MFCC coefficients from a dir of wav files')
    parser.add_argument('input_dir', help='The directory containing the audio to preprocess.')
    parser.add_argument('output_dir', help='The directory where MFCC coeficients should go.')
    parser.add_argument('plt_dir', help='optional, plot MFCC and save to separate plot directory')
    args = parser.parse_args()
    get_mfcc()
