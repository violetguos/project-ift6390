'''
File containing feature extraction methods.
Input should be the preprocessed version of the corpus.
'''

import argparse
import os
import subprocess
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
import matplotlib.pyplot as plt

from python_speech_features import mfcc
from python_speech_features import logfbank

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

def extract_mfccs(input_dir, output_dir):
    # Extract mfccs for all wav files
    for i, filename in enumerate(os.listdir(input_dir)):
    	if filename.endswith('.wav'):
    		input_file_path = os.path.join(input_dir, filename)
    		(rate,sig) = wav.read(input_file_path)
    		mfcc_feat = mfcc(sig,rate, nfft=551)
    		print("Saving MFCC text for {}\n".format(filename))
    		np.savetxt(os.path.join(output_dir, '{}.txt'.format(os.path.splitext(filename)[0])), mfcc_feat, delimiter=",")

def extract_formants(input_dir, output_dir):
	# This function requires the installation of Praat (see readme).
	for i, filename in enumerate(os.listdir(input_dir)):
		if filename.endswith('.wav'):
			print('Extracting formants for {} ({} of {})'.format(filename, i, len(os.listdir(input_dir))))
			filename_prefix = filename.split('.')[0]
			input_file_path = os.path.join(input_dir, filename)
			output_file_path = os.path.join(output_dir, '{}.txt'.format(filename_prefix))
			process = subprocess.Popen(['praat', 'extract_formants.praat', input_file_path, output_file_path])
			process.communicate()

def extract_spectrogram(input_dir, output_dir):
	pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extract features from the corpus data.')
	parser.add_argument('feature_type', help='The type of features you want to extract. Can be mfcc, formants, or spectrogram.')
	parser.add_argument('input_dir', help='The directory containing the data to extract features from.')
	parser.add_argument('output_dir', help='The directory where the extracted features should go.')
	args = parser.parse_args()

	if args.feature_type == 'mfcc':
		extract_mfccs(args.input_dir, args.output_dir)
	elif args.feature_type == 'formants':
		extract_formants(args.input_dir, args.output_dir)
	elif args.feature_type == 'spectrogram':
		extract_spectrogram(args.input_dir, args.output_dir)
	else:
		raise ValueError('Invalid feature type. Valid features to extract are: mfcc, formants, spectrogram.')