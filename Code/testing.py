import os
import librosa
import pyaudio
import pydub
import numpy as np

def load_audio(filename, sample_rate=22050, dtype=np.float32) -> np.ndarray:
    
    dtype = np.dtype(dtype)
    wave, _ = librosa.load(filename, sr=sample_rate, mono=True, dtype=dtype)
    return wave

a,_ = librosa.load("blues.00000.wav", sr = 22050,mono=True, dtype =np.dtype(np.float32))

print(a.shape)