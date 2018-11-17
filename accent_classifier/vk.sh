#!/bin/sh
#chmod +x vk.sh
# run the above ^ command on your own computer
# if you trust me ( ͡~ ͜ʖ ͡°)

# violet's testing shell script
# lastname is kuo, not joe


# !! always add the trailing `/` symbol!!!

# MFCC for American
#python mfcc_process.py /Users/vikuo/Documents/accent-classification-corpora/librispeech_preprocessed/ \
#/Users/vikuo/Documents/accent-classification-corpora/us_mfcc_coefs/ \
#/Users/vikuo/Documents/accent-classification-corpora/us_mfcc_plots/ \

# MFCC for British
python mfcc_process.py /Users/vikuo/Documents/accent-classification-corpora/librit_preprocessed/ \
/Users/vikuo/Documents/accent-classification-corpora/brit_mfcc_coefs/ \
/Users/vikuo/Documents/accent-classification-corpora/brit_mfcc_plots/ \
