
# IFT 6390 Project: Accent Classification
## How to structure python

* [overall practice]https://docs.python-guide.org/writing/structure/
* [what to install ]http://web.stanford.edu/class/cs224n/assignment1/index.html
  * tldr version:
  ```
  cd assignment1
  sudo pip install virtualenv      # This may already be installed
  virtualenv .env                  # Create a virtual environment
  source .env/bin/activate         # Activate the virtual environment
  pip install -r requirements.txt  # Install dependencies
  # Work on the assignment for a while ...
  deactivate                       # Exit the virtual environment
  ```
* [how to tell other ppl what to install ]https://medium.com/python-pandemonium/better-python-dependency-and-package-management-b5d8ea29dff1

## Administrative to-do's
* Get everyone invited to and on this Git repo. For those new to Git, check out:
    * This Git and GitHub basics [tutorial](https://www.elegantthemes.com/blog/resources/git-and-github-a-beginners-guide-for-complete-newbies)
    * [GitKraken](https://www.gitkraken.com/), a Git GUI (easy way to avoid the complications of the command line -- push, pull, commit, and undo mistakes easily)
* Check out [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) (joint editing of Jupyter notebooks), for those who want to develop using a notebook
* For everyone who wants to visualize a spectrogram, but especially for anybody working on formant extraction: install [Praat](http://www.fon.hum.uva.nl/praat/) and have a look at a [tutorial](https://www.gouskova.com/2016/09/03/praat-tutorial/)
* All should skim the papers pinned on Slack for a general understanding

## Game plan

### ~~Decide on accents and corpora to compare~~
* For North American style English: [Librispeech](http://www.openslr.org/12/)
* For British English: [Audio BNC](http://www.phon.ox.ac.uk/AudioBNC)

Corpora need to have enough data (on the order of a few hours), and contain audio of only one speaker at a time (not conversations or interviews where we also hear the interviewer). If we find another corpus meeting these requirements for another English, we can add another class.

### Preprocess and clean the corpus data
* We need to assemble recordings from Audio BNC consisting of single speakers. Look for tags such as "lecture" or "sermon".
* Audio BNC is in mono, while Librispeech is in stereo, so Librispeech needs to be downsampled to mono to match.
* Both corpora need to be cut into clips (a minute each? Maybe less, we'll see). There are scripts to do this sort of thing in Praat and with other Python libraries.

### Feature extraction
For MFCC features:
* Write a function that extracts MFCCs. There are existing functions for this, see [python speech features](https://python-speech-features.readthedocs.io/en/latest/) or [librosa](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html).

For F1, F2, F3 features:
* We need to decide which frames in the audio are _[voiced](https://en.wikipedia.org/wiki/Voice_(phonetics))_ (and thus will have good formants to measure). Deshpande et al. do this heuristically by counting a frame as voiced if the [log energy](https://python-speech-features.readthedocs.io/en/latest/#python_speech_features.base.logfbank) is greater than or equal to -9.0 and if the [zero crossing](https://en.wikipedia.org/wiki/Zero_crossing) is between 1 and 45. There are ways to do this in Python and/or Praat.
* Then we need to actually extract F1, F2, and F3 from each voiced frame. This is scriptable in Praat.

For raw spectrogram features:
* We should use [scipy's function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html) to get the spectrogram itself.

### Network building
We should use [Keras](https://keras.io/) because it's easy! First though, we should train a baseline SVM from sklearn.
For the RNN (LSTM, probably):
* Use with _sequential_ versions of MFCCs and formant measures.
* What would be cool is if we could return the part of the input sequence that is decisive for one accent vs. the other (analogous to an embedding in "accent space"?). That would make a cool plot: if we have the transcripts as well, maybe we could pull what words are being said out and see what sounds typically British or American.

For the CNN:
* Use with _non-sequential_ (i.e. concatenated) versions of MFCCs and formant measures, and most importantly with raw spectrogram data
* Since we're using raw spectrogram data, we could potentially generate new spectrograms after, according to a certain accent

Hyperparameter tuning needs to be done on everything.

### Report
Gotta write everything up.

### Poster
Also gotta make everything into a poster.
