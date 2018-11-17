'''
File containing feature extraction methods.
Input should be the preprocessed version of the corpus.
'''

import argparse
import os
import subprocess

def extract_mfccs(input_dir, output_dir):
	pass

def extract_formants(input_dir, output_dir):
	# This function requires the installation of Praat (see readme).
	for i, filename in enumerate(os.listdir(input_dir)):
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