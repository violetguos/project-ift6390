

# IFT 6390 Project: Accent Classification
## How to structure python

* [Overall practice](https://docs.python-guide.org/writing/structure/)
* [What to install ](http://web.stanford.edu/class/cs224n/assignment1/index.html)

  * TLDR Conda version:
  ```
  cd project
  # check that you installed conda
  conda create -n yourenv python=3.6
  source activate /insert/yourenv/name       # Activate the virtual environment
  pip install -r requirements.txt  # Install dependencies
  # Work on the assignment for a while ...
  source deactivate                       # Exit the virtual environment
  ```
  * tldr it also works version:
  ```
  cd assignment1
  sudo pip install virtualenv      # This may already be installed
  virtualenv .env                  # Create a virtual environment
  source .env/bin/activate         # Activate the virtual environment
  pip install -r requirements.txt  # Install dependencies
  # Work on the assignment for a while ...
  deactivate                       # Exit the virtual environment
  ```
* [How to tell other ppl what to install ](https://medium.com/python-pandemonium/better-python-dependency-and-package-management-b5d8ea29dff1)

* Don't write [code that smells bad](https://en.wikipedia.org/wiki/Code_smell)!

## Administrative to-do's

The collective Google Colab notebook is now [here](https://colab.research.google.com/drive/1ejfZhiqM3Wg4w9ofw4-ncoJzi-Aa8kje?fbclid=IwAR3AHejpp0D1Ky9oitURAzbRypzsPsExShGQzo5qzVftd5-w2naOkxp0cYU#scrollTo=d8wDXAOAbsUd). Test as you please! (Instructions on how to interact with Google Drive data coming.)
* ~~Get everyone invited to and on this Git repo. For those new to Git, check out:~~
    * This Git and GitHub basics [tutorial](https://www.elegantthemes.com/blog/resources/git-and-github-a-beginners-guide-for-complete-newbies)
    * [GitKraken](https://www.gitkraken.com/), a Git GUI (easy way to avoid the complications of the command line -- push, pull, commit, and undo mistakes easily)
* ~~Check out [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) (joint editing of Jupyter notebooks), for those who want to develop using a notebook~~
* For everyone who wants to visualize a spectrogram, but especially for anybody working on formant extraction: install [Praat](http://www.fon.hum.uva.nl/praat/) and have a look at a [tutorial](https://www.gouskova.com/2016/09/03/praat-tutorial/)
* All should skim the papers pinned on Slack for a general understanding

## Game plan

### ~~Decide on accents and corpora to compare~~
* For North American style English: [Librispeech](http://www.openslr.org/12/)
* ~~For British English: [Audio BNC](http://www.phon.ox.ac.uk/AudioBNC)~~
* Audio BNC did not end up working, so for British English, we assembled "Librit", a corpus of British audiobook narrators from [Librivox](https://librivox.org/) (actually the same source as Librispeech). Thanks to [RuthieG](https://golding.wordpress.com/home/other-british-readers-on-librivox/) for her list of British readers.

Corpora need to have enough data (on the order of a few hours), and contain audio of only one speaker at a time (not conversations or interviews where we also hear the interviewer). If we find another corpus meeting these requirements for another English, we can add another class.

### ~~Preprocess and clean the corpus data~~
* Both corpora are downsampled to mono audio.
* Both corpora are cut into clips (~~a minute each? Maybe less, we'll see~~ 20 seconds each). 

### Feature extraction
For MFCC features:
* Write a function that extracts MFCCs. There are existing functions for this, see [python speech features](https://python-speech-features.readthedocs.io/en/latest/) or [librosa](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html).

For F1, F2, F3 features:
* We need to decide which frames in the audio are _[voiced](https://en.wikipedia.org/wiki/Voice_(phonetics))_ (and thus will have good formants to measure). Deshpande et al. do this heuristically by counting a frame as voiced if the [log energy](https://python-speech-features.readthedocs.io/en/latest/#python_speech_features.base.logfbank) is greater than or equal to -9.0 and if the [zero crossing rate](https://en.wikipedia.org/wiki/Zero_crossing) is between 1 and 45. There are ways to do this in Python and/or Praat.
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
Gotta write everything up. Violet has a NIPS template [here](https://www.overleaf.com/6314387546jfjzstdczzpm).

### Poster
Also gotta make everything into a poster.
