import os
import librosa
import pyaudio
import pydub
import numpy as np


floating_dtypes = [np.float16, np.float32, np.float64]

integer_dtypes = [np.int8, np.int16, np.int32, np.int64]


def play(array):
    audio = pyaudio.PyAudio()

    
    stream = audio.open(format=pyaudio.paFloat32, channels=1, rate= 22050, output=True)
    stream.write(array.tobytes())
    stream.stop_stream()
    stream.close()
    audio.terminate()