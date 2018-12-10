'''
Functions to get a plot of which elements are
most important in the NN's decision.
'''

import numpy as np
import argparse
import os
import pickle
import copy
import matplotlib.pyplot as plt
import sys

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 20,1.5

from keras.models import load_model

def get_prob_diffs(expanded_sequence, length, model):
	# One by one, perturb each element in the sequence.
	# Then predict the probabilities.
	# The segments that change the probs the most when perturbed are
	# the most "influential" on the classification.
	base_probs = model.predict(expanded_sequence)[0]
	prob_diffs = []

	for i in range(0, length):
		perturbed_sequence = copy.deepcopy(expanded_sequence)
		perturbed_sequence[0][i] += 0.5
		new_probs = model.predict(perturbed_sequence)[0]
		prob_diff = np.linalg.norm(base_probs - new_probs)
		pair = [i, prob_diff]
		if i % 50 == 0:
			print('Pair {}: {}'.format(i, pair))
		prob_diffs.append(pair)

	prob_diffs = np.array(prob_diffs)
	return prob_diffs

def get_plot(prob_diffs, length, output_dir, filename_prefix):
	x = np.arange(length)
	y = prob_diffs[:,1]
	y = y.astype('float')
	y[y == 0] = 0.001 	# To avoid taking the log of 0
	y = -1 * np.log(y) 	# To make values more contrastive

	if length == 1999:
		plt.xticks(np.arange(0, length, step=100))
	elif length == 400:
		x = x * 5
		plt.xticks(np.arange(0, 2000, step=100))

	extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
	plt.imshow(y[np.newaxis,:], cmap="viridis", aspect="auto", extent=extent)

	
	plt.xlabel('Time (ms)')
	plt.yticks([])
	plt.xlim(extent[0], extent[1])
	#plt.xticks(np.arange(0, 2000, step=100))
	plt.tight_layout()

	plt.savefig(os.path.join(output_dir, '{}_plot.png'.format(filename_prefix)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get a plot of which elements are most important in classification of an input file.')
	parser.add_argument('feature_type', help='The type of feature of your input file. Can be mfcc, formants, or spectrogram.')
	parser.add_argument('input_file', help='The path to the input file.')
	parser.add_argument('model_path', help='The path to the trained model.')
	parser.add_argument('output_dir', help='The path to the output directory.')
	parser.add_argument('--plotonly', help='Gets the plot only. (Needs the already-calculated dist matrix to be present in the output_dir.', action="store_true")
	args = parser.parse_args()

	# Set up sequence length (magic numbers from preprocessing)
	if args.feature_type == 'mfcc':
		length = 1999
	elif args.feature_type == 'formants':
		length = 400
	else:
		raise ValueError('Invalid feature type. Valid features types are: mfcc, formants, spectrogram.')

	# Import the filename, model and the pickle (if it exists already)
	model = load_model(args.model_path)
	_, filename = os.path.split(args.input_file)
	filename_prefix = filename.split('.')[0]
	pickle_path = os.path.join(args.output_dir, '{}_probdiff.pickle'.format(filename_prefix))

	if not args.plotonly:
		# Read the input file
		with open(args.input_file, 'r') as fp:
			lines = fp.readlines()
		sequence = []
		for line in lines:
			point = line.split(',')
			point = [float(x) for x in point]
			sequence.append(point)
		sequence = np.array(sequence)

		# Since the model was trained on a minibatch, we need to still
		# respect that dimension, even though we're just going to predict one point.
		expanded_sequence = np.expand_dims(sequence, axis=0)

		# Get the prob diffs
		prob_diffs = get_prob_diffs(expanded_sequence, length, model)
		with open(pickle_path, 'wb') as jar:
			pickle.dump(prob_diffs, jar, protocol=pickle.HIGHEST_PROTOCOL)

	else:
		# Pull up the pickle
		with open(pickle_path, 'rb') as jar:
  			prob_diffs = pickle.load(jar)
		
	# Get the plot
	get_plot(prob_diffs, length, args.output_dir, filename_prefix)