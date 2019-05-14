#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 23:33:56 2019

@author: kronert
"""
# We change the working directory
import os
os.chdir('../')

# Import librairies
import pandas as pd
import numpy as np
from zipfile import ZipFile 
from app.ML.src.Caracters import Caracters


#Hyperparameters
nbLigns = 10000


# Get the Inputs
archive = ZipFile('Data/raw_data/fra-eng.zip')
data = pd.read_csv(archive.open('fra.txt'),
                   sep='\t',header=None, names=['english','french'])[:nbLigns]

## Preprocessing
data['french'] = data["french"].apply(lambda x : '\t' + x + '\n')

# load
eng_car = Caracters()
fra_car = Caracters()


eng_car.load('app/ML/save/eng_car.pickle')
fra_car.load('app/ML/save/fra_car.pickle')


print(eng_car.voc_size)


# Predict
input_encoder = eng_car.predict(list(data['english'].values))
input_decoder = fra_car.predict(list(data['french'].values))
output_decoder = np.zeros(input_decoder.shape)
output_decoder[:,:-1,:] = input_decoder[:,1:,:]
# save data
np.savez_compressed('Data/processed_data/vectorised_fra_eng',
                    input_encoder=input_encoder,
                    input_decoder=input_decoder,
                    output_decoder=output_decoder
                   )


