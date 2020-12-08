import os
import librosa
import pyaudio
import pydub
import numpy as np
import pandas as pd
dir = "./testing"

def load_audio(filename, sample_rate=22050, dtype=np.float32) -> np.ndarray:
    
    dtype = np.dtype(dtype)
    wave, _ = librosa.load(filename, sr=sample_rate, mono=True, dtype=dtype)
    return wave


def gather_data(path):
    music =[]
    for dir0 in os.listdir(path):
        
        full_path=os.path.join(path, dir0) 
        
        if os.path.isdir(full_path):
            print(dir0)
            for file in os.listdir(full_path):
                
                music_path=os.path.join(full_path, file) 
                music.append(load_audio(music_path))
                
    return np.array(music)        

a = gather_data("./genres")
print(a.shape)