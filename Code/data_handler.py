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


def split_audio(path):
    

    #for dir0 in os.listdir(path):
        
        #full_path=os.path.join(path, dir0) 
        
        #if os.path.isdir(full_path):
        #    print(dir0)
            
        for file in os.listdir(path):
            
            music_path=os.path.join(path, file)
            
            for i in range(10):
                t1 = 3*(i)*1000
                t1 = 3*(i+1)*1000
                audio = AudioSegment.from_wav(music_path)
                audio.export(file[:len(file)-4]+str(i)+".wav" ,format='wav')

def create_data_img(path):
    
        
    for file in os.listdir(path):
        
        music_path=os.path.join(path, file)
        
        y, sr = load_audio(music_path)
        mels = librosa.feature.melspectrogram(y = y, sr =sr)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels,ref = np.max))
        plt.savefig(file+".png")
                
def gather_data_img(path):
    
    data = []
       
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = Image.open(img_path)
        img = np.array(img)
        data.append(img)
    return np.array(data)


def get_classes_img():
    classes = []
    n = 0
    for i in range(1000):
        ohe = np.zeros(10)
        if i % 100 == 0 and i != 0:
            n +=1
        ohe[n] = 1
        classes.append(ohe)
    return np.array(classes)

#create_data_img("./genres_img")