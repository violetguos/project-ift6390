#!/usr/bin/env bash





# to re-slice the original audio into small time segments
# run python timeSlicePickle.py [copora name: 'librispeech' or 'librit']  \
# [clip_length: in terms of seconds]

# after slicing into small time segments, pickle the sepctrograms of each
# run python spectrogramPickle.py \
# ['pickle' : re-pickle all the spectrogram for train and test]
# ['test': test loading pickle and verify shapes]
# ['plot': plot spectrogram in the freqeuncy scale]


# after training and saving the .h5 model, run
# python heatmap.py specto_hd_cnn_hyperTuned.h5
