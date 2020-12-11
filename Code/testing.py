jon_hace_la_encuesta = True

import numpy as np

a =  np.array([0,1,0,0])
b = np.array([1,0,0,0])

print(np.sum(np.abs(a-b)))