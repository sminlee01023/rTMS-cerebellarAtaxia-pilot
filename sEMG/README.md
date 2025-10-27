### Preprocessing and Feature Extraction for surface EMG(Delsys, US) data


In this study, surface EMG (sEMG) data were acquired by placing electrodes on both the medial and lateral parts of the Gastrocnemius muscles (GM) on both legs.  
This bilateral placement allows detailed assessment of muscle activity pattern variations during experimental tasks.

<br/>
Please note, due to privacy and confidentiality concerns, the raw sEMG data cannot be shared publicly.

<br/><br/>

#### EMG Data Preprocessing
The raw EMG signals were first centered by removing the mean to eliminate baseline offset. A bandpass filter (10–500 Hz) was applied to isolate the physiological frequency range of the muscle signals while removing low- and high-frequency noise. Next, a notch filter was used to eliminate specific frequency interference (e.g., 60 Hz powerline noise).

The filtered signal underwent detrending to remove linear trends followed by full-wave rectification, converting all signal values to positive for further analysis. The root mean square (RMS) envelope was computed using a sliding window method (window size: 250 samples) to quantify the signal amplitude over time.

A dynamic threshold based on the mean plus standard deviation of the RMS envelope was calculated to detect significant muscle activations. If no activation points were detected initially, the threshold was reduced by half for sensitivity.

<br/><br/>

#### Usage

The preprocessing code can be run by importing the module `EMGprep` and using the `process_emg_files` function with appropriate base directories and category labels. The function iterates over all subject files, extracts required features, and returns summarized results ready for visualization or statistical testing.

```python
from EMGprep import process_emg_files

process_emg_files(base_dir, ['01_PRE', '02_POST', '03_FOLLOW'], num_subjects=10)
```
