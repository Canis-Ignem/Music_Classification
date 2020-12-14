import os
import librosa
import pyaudio
import pydub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from pydub import AudioSegment
min = 660000

def load_audio(filename, sample_rate=22050, dtype=np.float32) -> np.ndarray:
    
    dtype = np.dtype(dtype)
    wave, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=dtype)
    return wave, sr


def gather_data(path):

    music = []
    for dir0 in os.listdir(path):
        
        full_path=os.path.join(path, dir0) 
        
        if os.path.isdir(full_path):
            print(dir0)
            
            for file in os.listdir(full_path):
                
                music_path=os.path.join(full_path, file)
                y, _ = load_audio(music_path) 
                music.append(y)
            
    return np.array(music)  

#a = gather_data("./genres")


def format_data(data):
    splitedData = []
    classes = []
    n = 0
    
    mod = []
    for i in data:
        i = i[:660000]
        a = np.split(i,100)
        for j in a:
            mod.append(j)
    mod = np.array(mod)
    for i in range(mod.shape[0]):
        ohe = np.zeros(10)
        if i % 10000 == 0 and i != 0:
            n +=1
        wav = mod[i]
        wavList = np.split(wav, 60)
        splitedData.append(wavList)
        ohe[n] = 1
        classes.append(ohe)
    return np.array(splitedData), np.array(classes)


def get_batch(data,classes, batch_size = 100):
   indices = np.random.permutation(data.shape[0])[0:batch_size]
   return data[indices] , classes[indices]


# 258 430
def create_data_mels(path):
    data = np.zeros((3000,128,430))
    classes = []
    n = -1
    i = 0
    for dir0 in os.listdir(path):

        full_path=os.path.join(path, dir0)

        if os.path.isdir(full_path):
            print(dir0)
            n += 1
            for file in os.listdir(full_path):
                ohe = np.zeros(10)
                ohe[n] = 1
                for j in range(24):
                    classes.append(ohe)
                
                music_path=os.path.join(full_path, file)
                
                y, sr = load_audio(music_path)
                mels = librosa.feature.melspectrogram(y = y, sr =sr)
                #print(mels.shape)
                mels = mels[:,:1290]
                '''
                #mels = np.reshape(mels, (256,645))
                data[i,:,:] = mels
                i += 1
                '''
                data[i,:,:] = mels[:,:430]
                i += 1
                data[i,:,:] = mels[:,430:860]
                i += 1
                data[i,:,:] = mels[:,860:]
                i += 1
                
                '''
                a = mels[:,:430]
                data[i,:,:] = a
                i += 1
                for i in range(1,2):
                    b = mels[:,430*i:430*(i+1)]
                    data[i,:,:] = b
                    i += 1
                data[i,:,:] = mels[:,430*2:]
                '''
    data = np.reshape(data,(24000,16,430))
    return data,np.array(classes)

