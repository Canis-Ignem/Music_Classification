jon_hace_la_encuesta = True
import data_handler as dh
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
def load_audio(filename, sample_rate=22050, dtype=np.float32) -> np.ndarray:
    
    dtype = np.dtype(dtype)
    wave, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=dtype)
    return wave, sr


y, sr = load_audio("./blues.00000.wav")

mels = librosa.feature.melspectrogram(y = y, sr =sr)

fig = plt.Figure()

canvas = FigureCanvas(fig)

p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
plt.savefig("test.png")
