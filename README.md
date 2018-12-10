
# IFT6390 Final Project: Speaker Accent Classification with Deep Learning

Team members:

* Jonathan Bhimani-Burrows (20138260)
* Khalil Bibi (20113693)
* Arlie Coles (20121051)
* Akila Jeeson Daniel (20140681)
* Y. Violet Guo (20120727)
* Louis-François Préville-Ratelle (708048)


## File structure for submission


**Note:** Our original audio corpora are not stored in this repository (due to their size), but we will supply it upon request.

Our trained neural network models can be found in `models/`.

Source code can be found in `accent-classifier/`.

Experiment notebooks can be found in `experiments/` (though we do not recommend re-running these).

## Example

Under `supplementaryMaterial/`, we include `1_0_20sec.wav`, a 20 second file from Librit (our British audio corpus). In our report, we specify that our neural networks identify the following:

| Phoneme | Word of the phoneme | Time Window in Audio (seconds) |
| :-----: | :-----------------: | :----------------------------: |
|    *eu*     |   *F**eu**erbach*   |          1:49 to 2:19          |
|   **or**      |      **or**g      |         10:86 to 11:39         |
|    **eu**     |   *F**eu**erbach*   |         17:65 to 18:41         |

You can play the audio  using any open source or commercial audio player to skip to the following timestamps and verify our results. Additionally, the image files showing the visualization of the weights learned by the neural network are available here in higher resolution.


## Installation

First, clone the repository:
```
git clone https://github.com/violetguos/project-ift6390.git 
cd project-ift6390
```

We recommend running any code in a virtual environment. This allows for easy installation of dependencies:
 
  ```
  sudo pip install virtualenv      # This may already be installed
  virtualenv .env                  # Create a virtual environment
  source .env/bin/activate         # Activate the virtual environment
  pip install -r requirements.txt  # Install dependencies
  # Run any code you need...
  deactivate                       # Exit the virtual environment
  ```
A special dependency needed (for formant feature extraction) is the acoustic software  [Praat](http://www.fon.hum.uva.nl/praat/), which needs to be added to your system's path ([Mac](https://www.architectryan.com/2012/10/02/add-to-the-path-on-mac-os-x-mountain-lion/)/[Windows](https://www.itprotoday.com/cloud-computing/how-can-i-add-new-folder-my-system-path)/[Unix](https://stackoverflow.com/questions/14637979/how-to-permanently-set-path-on-linux-unix)).
