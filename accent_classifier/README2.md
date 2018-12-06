# IFT6390 Final Project

## Team Members

* Jonathan Bhimani-Burrows (20138260)
* Khalil Bibi (20113693)
* Arlie Coles (20121051)
* Akila Jeeson Daniel (20140681)
* Y. Violet Guo (20120727)
* Louis-François Préville-Ratelle (708048)

# Code files and explanations on how to run them

Our `.py` files in `./accent_classifier` are used for preprocessing and pickling, whereas the `.ipynbs` are used for the neural network trainings

# example file structure for submission
`.
├── report.pdf
├── README.md
├──
├── accent_classifier
│   ├── README2.md
│   ├──
`


# File Structure



# Details of each
* `extract_formants.praat`: script written in Praat to extract formants
* `preprocessing.py`: chop long audio files into small chunks, noise removal, and stereo to mono
* `feature_extraction.py`: python script to extract MFCC, formants (by calling `extract_formants.praat` in the python script) and spectrogram
* `extras.py`: Functions to get a plot of which elements are most important in the NN's decision. Used in RNN feature map analysis section of the report.
* `heatmap.py`: we load a trained model in .h5 format, and pass an example image through the network
to find the hidden layer weights visualizations
* `timeSlicePickle.py`: to re-slice audio into small time segments, used by CNN, see report for the detailed reasons
* `spectrogramPickle.py`: final pickle pulls the 'spectrum' key of all
 the `.1sec_hdpickles` and aggregates them

* `Wav_2_Spectrogram.ipynb`: experiment with spectrogram
