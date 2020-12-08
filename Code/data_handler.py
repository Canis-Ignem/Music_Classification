import os
import librosa
import pyaudio
import pydub
import numpy as np
import pandas as pd

min = 660000

def load_audio(filename, sample_rate=22050, dtype=np.float32) -> np.ndarray:
    
    dtype = np.dtype(dtype)
    wave, _ = librosa.load(filename, sr=sample_rate, mono=True, dtype=dtype)
    return wave


def gather_data(path):

    #df = pd.DataFrame(columns=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"])
    music = []
    for dir0 in os.listdir(path):
        
        full_path=os.path.join(path, dir0) 
        
        if os.path.isdir(full_path):
            print(dir0)
            
            for file in os.listdir(full_path):
                
                music_path=os.path.join(full_path, file) 
                music.append(load_audio(music_path))
            
    return np.array(music)  

#a = gather_data("./genres")


def format_data(data):
    splitedData = []
    classes = []
    n = 0
    
    for i in range(data.shape[0]):
        ohe = np.zeros(10)
        if i % 100 == 0 and i != 0:
            n +=1
        wav = data[i][:min]
        wavList = np.split(wav, 30)
        splitedData.append(wavList)
        ohe[n] = 1
        classes.append(ohe)
    return np.array(splitedData), np.array(classes)

#b, c = format_data(a)
#print(b.shape, c.shape)

def get_batch(data,classes, batch_size = 100):
   indices = np.random.permutation(data.shape[0])[0:batch_size]
   return data[indices] , classes[indices]

#d, e = get_batch(b,c)
#print(d.shape, e.shape)

