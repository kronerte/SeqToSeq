#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:39:43 2019

@author: kronert
"""
import os
os.chdir('../')

# Import librairies
import pandas as pd
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



## Fit
eng_car = Caracters()
eng_car.fit(data['english'])

fra_car = Caracters()
fra_car.fit(data['french'])

## Save Models
eng_car.save('app/ML/save/eng_car.pickle')
fra_car.save('app/ML/save/fra_car.pickle')





