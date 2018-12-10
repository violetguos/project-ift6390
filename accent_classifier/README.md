
# IFT 6390 Project: Accent Classification


## Source code documentation

### Preprocessing and feature extraction

Files for preprocessing and feature extraction can be found under `preprocessing/`.

The preprocessing methods can be found in `preprocessing.py`. They convert all corpus audio to segments of a given length, and sample everything to mono 16,000 KHz `.wav` files. To run the preprocessing, run:

```python preprocessing.py [corpus] [input_dir] [output_dir] [segment length (in seconds)] ```

Possible `[corpus]` arguments are: `librispeech`, `librit`.

The feature extraction methods can be found in `feature_extraction.py`.  To run feature extraction for a given feature, run:

```python feature_extraction.py [feature type] [input_dir] [output_dir]```

Possible `[feature_type]` arguments are: `mfcc`, `formants`, `spectrogram`.

Other files included here are:
  * `extract_formants.praat`: Praat script to extract formants
  * `timeSlicePickle.py`: to re-slice audio into small time segments, used by CNN, see report for the detailed reasons
  * `spectrogramPickle.py`: final pickle pulls the 'spectrum' key of all
 the `.1sec_hdpickles` and aggregates them
  * `Wav_2_Spectrogram.ipynb`: experiment with spectrogram

### Postprocessing/visualization

The files under `postprocessing/` are used to help visualize our learned neural network weights.

The visualization methods for RNNs are in `extras.py`. To generate a visualization, run:

```python extras.py [feature_type] [input_feature_path] [model_path] [output_dir] [--plotonly]```

Possible `[feature_type]` arguments are: `mfcc`, `formants`, `spectrogram`. Use the ``--plotonly`` switch to only generate the plot (without recalculating any weight purturbations).

The visualization methods for CNNs are in `heatmap.py`.  This uses the `keras-vis` API to visualize the CNN filters, given the `.h5` model saved. We load a trained model in and pass an example image through the network to generate the hidden layer weight visualizations.
