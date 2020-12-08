import os
import librosa
import pyaudio
import pydub
import numpy as np

dir = "./testing"

def load_audio(filename, sample_rate=22050, dtype=np.float32) -> np.ndarray:
    
    dtype = np.dtype(dtype)
    wave, _ = librosa.load(filename, sr=sample_rate, mono=True, dtype=dtype)
    return wave




vectors = []
for file in os.scandir(dir):
    if file.path.endswith(".wav"):
        a,_ = librosa.load(file.path, sr = 22050,mono=True, dtype =np.dtype(np.float32))
        vectors.append(a)


print(vectors[0][:5])






