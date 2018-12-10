# IFT6390 Final Project

## Team Members

* Jonathan Bhimani-Burrows (20138260)
* Khalil Bibi (20113693)
* Arlie Coles (20121051)
* Akila Jeeson Daniel (20140681)
* Y. Violet Guo (20120727)
* Louis-François Préville-Ratelle (708048)

# Code files and explanations on how to run them

## file structure for submission
```
.
├── README.md
├── cnn_log.txt
├── formant_rnn
│   └── Formants\ RNN.ipynb
├── mfcc_rnn
├── pickleScript.sh
├── postprocessing
│   ├── extras.py
│   └── heatmap.py
├── preprocessing
│   ├── Wav_2_Spectrogram.ipynb
│   ├── extract_formants.praat
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── spectrogramPickle.py
│   └── timeSlicePickle.py
├── spec_cnn
│   ├── CNN\ hyperparams\ tune\ project.ipynb
│   ├── final_cnn.ipynb
│   ├── specto_hd_cnn_hyperTuned.h5
│   └── spectrogram_svm.ipynb
└── supplementaryMaterial
    ├── 1_0_20sec.wav
    ├── cnn_filter_conv2d_10_19sec_normalized_flipped.npz
    ├── cnn_filter_conv2d_10_19sec_normalized_flipped.png
    ├── cnn_filter_conv2d_11_19sec_normalized_flipped.npz
    ├── cnn_filter_conv2d_11_19sec_normalized_flipped.png
    ├── cnn_filter_conv2d_12_19sec_normalized_flipped.npz
    ├── cnn_filter_conv2d_12_19sec_normalized_flipped.png
    ├── overlayed_conv2d_10_19sec.png
    ├── overlayed_conv2d_11_19sec.png
    ├── overlayed_conv2d_12_19sec.png
    └── saved_spectrogram_20_sec_wav.png
```

# Details of each file

## Important note
Our data files are impossible to upload, therefore we will supply data upon request only.

## Code and other material

* `formant_rnn/`: contains the notebook that runs Formant RNN, `Formants\ RNN.ipynb`. Load locally or on google colab.

* `mfcc_rnn/`: contains the notebook that runs MFCC RNN.

* `spec_cnn`: contains all code relating to CNN training, testing, validation.
  * `CNN\ hyperparams\ tune\ project.ipynb`: GridSearch for the CNN's hyperparameters
  * `final_cnn.ipynb`: the final Spec-CNN using our grid search tuned hyperparameters
  *  `specto_hd_cnn_hyperTuned.h5`: the saved weights of the final CNN model
  * `spectrogram_svm.ipynb`: SVM and MF dummy classifier baselines.

* `supplementaryMaterial`: this contains the 20 second British Audio file. In our report, we have specified that our NNs learned the following:

| Phoneme | Word of the phoneme | Time Window in Audio (seconds) |
| :-----: | :-----------------: | :----------------------------: |
|    *eu*     |   *F**eu**erbach*   |          1:49 to 2:19          |
|   **or**      |      **or**g      |         10:86 to 11:39         |
|    **eu**     |   *F**eu**erbach*   |         17:65 to 18:41         |

You may play the audio `1_0_20sec.wav` using any open source or commercial audio player to skip to the following timestamps and verify our results.

* `postprocessing/`: used to visualize our RNN weights.
  * `extras.py` includes our original implementation of the RNN visualization.  Functions to get a plot of which elements are most important in the NN's decision. Used in RNN feature map analysis section of the report.
  * `heatmap.py`: using Keras-vis API, visualization of CNN filters, given the `.h5`
  model saved. we load a trained model in .h5 format, and pass an example image through the network
  to find the hidden layer weights visualizations

* `preprocessing/`:
  * `extract_formants.praat`: script written in Praat to extract formants
  * `preprocessing.py`: chop long audio files into small chunks, noise removal, and stereo to mono
  * `feature_extraction.py`: python script to extract MFCC, formants (by calling `extract_formants.praat` in the python script) and spectrogram

  * `timeSlicePickle.py`: to re-slice audio into small time segments, used by CNN, see report for the detailed reasons
  * `spectrogramPickle.py`: final pickle pulls the 'spectrum' key of all
 the `.1sec_hdpickles` and aggregates them
  * `Wav_2_Spectrogram.ipynb`: experiment with spectrogram
