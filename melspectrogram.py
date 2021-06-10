import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt

n_mfcc = 13
n_mels = 40
n_fft = 512
hop_length = 160
fmin = 0
fmax = None

mel_spect = librosa.feature.melspectrogram(y=signal, sr=fs, n_fft=n_fft,
                                     hop_length=hop_length,
                                     fmin=fmin, fmax=fmax, htk=False)

mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

librosa.display.specshow(mel_spect, y_axis='mel', fmax = fs, x_axis='time')

plt.show()
