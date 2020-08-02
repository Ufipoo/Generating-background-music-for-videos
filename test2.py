import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
song = np.load("seeds.npy")
i = 1
"""
while True:
    if np.load("JCP_mixed/classic_piano_train_" + str(i) + ".npy").any():
        song2 = np.load("JCP_mixed/classic_piano_train_" + str(i) + ".npy")
        print(song2)
    i += 1"""
directory = "D:/diplom/JCP_mixed"
files = os.listdir(directory)
print(files)
for i in range(0, len(files)):
    print(files[i])
    song = np.load("JCP_mixed/" + str(files[i]))
    print(np.shape(song))
    print(song[0][1])
    song1 = np.zeros((64,84,1))
    for j in range(np.shape(song)[0]):
        for k in range(np.shape(song)[1]):
            if song[j][k] == True:
                song1[j][k] = 1
                #song1[j][k] = np.reshape(song1[j],84)
                #print(song[j][k])
    print(np.reshape(song1,(64,84)))
    print(np.shape(song1))
    #print(song)
    print('_____________________________')