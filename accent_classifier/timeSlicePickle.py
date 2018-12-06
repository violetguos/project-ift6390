import os

import numpy as np

#!ls /Users/vikuo/Documents/accent-classification-corpora
librispeech_path = '/Users/vikuo/Documents/accent-classification-corpora/librispeech'
librispeech_preprocessed_path = '/Users/vikuo/Documents/accent-classification-corpora/librispeech_preprocessed'
librit_path = '/Users/vikuo/Documents/accent-classification-corpora/librit'
librit_preprocessed_path = '/Users/vikuo/Documents/accent-classification-corpora/librit_preprocessed'

import subprocess
import scipy.io.wavfile as wav
import sys
import os
import glob
import argparse
import numpy as np
import matplotlib
#matplotlib.use("TkAgg")
# Vi Comment: if i don't call this line, matplotlib crashes and gives me a call
# stack trace that triggers *literally and figuratively* me and my computer
import matplotlib.pyplot as plt
import pickle
from scipy.io.wavfile import read
from scipy import signal


from tqdm import tqdm


from functools import reduce
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks

import math
import argparse
import os

def remove_silence(audio):
	# Lightly adapted from:
	# https://stackoverflow.com/questions/23730796/using-pydub-to-chop-up-a-long-audio-file
	# We consider it silent if quieter than -16 dBFS for at least half a second.
	# (Might not use this if we want to match up with time stamps from transcriptions.)
	# Also it doesn't work right now anyway - debug later.
	audio_parts = split_on_silence(audio, min_silence_len=500, silence_thresh=-16)
	audio = reduce(lambda a, b: a + b, audio_parts)	# Re-combine
	return audio

def stereo_to_mono(audio):
	mono_audio = audio.set_channels(1)
	return mono_audio

def get_speaker_dict(input_dir, corpus):
	# Group together the files belonging to each speaker
	files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
	speaker_files = {}
	for file in files:
		if corpus == 'librispeech':
			speaker_id = file.split('-')[0]
		elif corpus == 'librit':
			speaker_id = file.split('_')[0]
		if speaker_id in speaker_files:
			speaker_files[speaker_id].append(file)
		else:
			speaker_files[speaker_id] = [file]
	return speaker_files

def downsample(audio, sample_rate):
	# Downsamples an audio object to a given rate
	audio = audio.set_frame_rate(sample_rate)
	return audio

def consolidate_speakers(input_dir, speaker_id, speaker_dict):
	# Concatenates all files from specific speaker into one audio object.
	file_list = speaker_dict[speaker_id]
	consolidated = AudioSegment.empty()
	#print_every = 10
	for i, file in enumerate(file_list):

		#print('	Processing file {} of {}...'.format(i, len(file_list)))
		file_audio = AudioSegment.from_wav(os.path.join(input_dir, file))
		consolidated += file_audio
	return consolidated

def preprocess_violet(input_dir, output_dir, length_in_sec, corpus):
	speaker_dict = get_speaker_dict(input_dir, corpus)
	speaker_ids = list(speaker_dict.keys())
	for j, speaker_id in enumerate(speaker_ids):
		print('Making files for speaker {} ({} of {})...'.format(speaker_id, j, len(speaker_ids)))
		# Get the concatenated audio for each speaker
		speaker_audio = consolidate_speakers(input_dir, speaker_id, speaker_dict)
		# Remove silence
		#speaker_audio = remove_silence(speaker_audio)
		# Downsample to mono
		speaker_audio = stereo_to_mono(speaker_audio)
		# Librit needs to be downsampled to 16k
		if corpus == 'librit':
			speaker_audio = downsample(speaker_audio, 16000)
		# Split into clips of length
		length = length_in_sec * 1000
		audio_chunks = make_chunks(speaker_audio, length)
		# Export the clips
		for i, chunk in enumerate(audio_chunks):
			chunk.export(os.path.join(output_dir, '{}_{}.wav'.format(speaker_id, i)), format='wav')


def preprocess(input_dir, output_dir, length_in_sec, corpus):
  speaker_dict = get_speaker_dict(input_dir, corpus)
  speaker_ids = list(speaker_dict.keys())


  file_count = 0
  for speaker_id in tqdm(speaker_ids):
    #print('Making files for speaker {} ({} of {})...'.format(speaker_id, j, len(speaker_ids)))
    # Get the concatenated audio for each speaker
    speaker_audio = consolidate_speakers(input_dir, speaker_id, speaker_dict)
    # Remove silence
    #speaker_audio = remove_silence(speaker_audio)
    # Downsample to mono
    speaker_audio = stereo_to_mono(speaker_audio)
    # Librit needs to be downsampled to 16k
    if corpus == 'librit':
      speaker_audio = downsample(speaker_audio, 16000)
    # Split into clips of length
    length = length_in_sec * 1000
    audio_chunks = make_chunks(speaker_audio, length)
    # Export the clips
    for i, chunk in enumerate(audio_chunks):
      file_count += 1
      chunk.export(os.path.join(output_dir, '{}_{}.wav'.format(speaker_id, i)), format='wav')
  print('file_count = ',file_count)




def extract_spectrogram_violet(input_dir, output_dir):
    cnt = 0
    print_every = 100

    # to fix the os.listdir having varing change
    os_list = os.listdir(input_dir)
    print("fixed os list: ", len(os_list))
    for i, filename in enumerate(os_list):
        cnt += 1
        if filename.endswith('.wav'):
            if cnt % print_every == 1:
                print('file number {}'.format(cnt))
                print('Getting spectrogram for {} ({} of {})'.format(filename, i, len(os.listdir(input_dir))))
            filename_prefix = filename.split('.')[0]
            input_file_path = os.path.join(input_dir, filename)
            x_value = 0
            sr_value = 0
            sr_value, x_value = read(input_file_path)
            img = 0

            # Get the spectrogram
            #spectrum, specs, t, img = plt.specgram(x_value, NFFT=80, Fs=16000, noverlap=40)
            spectrum, specs, t, img = plt.specgram(x_value, NFFT=400, Fs=16000, noverlap=160)
            #plt.ylim([0,2000])
            #plt.xlim([0,1])
            #plt.show()
            #print(spectrum.shape)
            plt.clf() # Clear plot - this is necessary!


            #stop

            # Dump to a dictionary
            spect_dict ={}
            #spect_dict['img'] = img
            spect_dict['t'] = t
            spect_dict['specs'] = specs
            spect_dict['spectrum'] = spectrum
            with open(os.path.join(output_dir, '{}.1sec_hdpickle'.format(filename_prefix)), "wb" ) as jar:
                pickle.dump(spect_dict ,jar,protocol=pickle.HIGHEST_PROTOCOL)
def extract_spectrogram(input_dir, output_dir):
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.wav'):
            #print('Getting spectrogram for {} ({} of {})'.format(filename, i, len(os.listdir(input_dir))))
            filename_prefix = filename.split('.')[0]
            input_file_path = os.path.join(input_dir, filename)
            x_value = 0
            sr_value = 0
            sr_value, x_value = read(input_file_path)
            img = 0

            # Get the spectrogram
            #spectrum, specs, t, img = plt.specgram(x_value, NFFT=80, Fs=16000, noverlap=40)
            spectrum, specs, t, img = plt.specgram(x_value, NFFT=400, Fs=16000, noverlap=160)
            #plt.ylim([0,2000])
            #plt.xlim([0,1])
            #plt.show()
            #print(spectrum.shape)
            plt.clf() # Clear plot - this is necessary!


            #stop

            # Dump to a dictionary
            spect_dict ={}
            #spect_dict['img'] = img
            spect_dict['t'] = t
            spect_dict['specs'] = specs
            spect_dict['spectrum'] = spectrum
            with open(os.path.join(output_dir, '{}.1sec_hdpickle'.format(filename_prefix)), "wb" ) as jar:
                pickle.dump(spect_dict ,jar,protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

	if sys.argv[1] == "librispeech" or sys.argv[1] == "librit":
		corpus = sys.argv[1]
		input_dir = "/Users/vikuo/Documents/accent-classification-corpora/"+corpus
		output_dir = "/Users/vikuo/Documents/accent-classification-corpora/"+corpus+"_preprocessed_167sec"

		print("input_dir", os.path.exists(input_dir))
		print("output_dir", os.path.exists(output_dir))
		clip_length = float(sys.argv[2])
		preprocess(input_dir, output_dir, clip_length, corpus)

		preprocessed_path = "/Users/vikuo/Documents/accent-classification-corpora/"+corpus+"_preprocessed_167sec"

		spectrogram_path = "/Users/vikuo/Documents/accent-classification-corpora/"+corpus+"_167sec_hdpickle"
		print("preprocessed_path", os.path.exists(preprocessed_path))
		print("spectrogram_path", os.path.exists(spectrogram_path))

		extract_spectrogram(preprocessed_path, spectrogram_path)
	else:
		print("error, corpus invalid!")
