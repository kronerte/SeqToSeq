#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:10:45 2019

@author: kronert
"""
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
class SequentialEncoder():
    def __init__(self, num_encoder_tokens, latent_dim):
       # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None, num_encoder_tokens))
        # LSTM
        self.encoder = LSTM(latent_dim, return_state=True)
        self.encoder_outputs, state_h, state_c = self.encoder(self.encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]
        
        self.model = Model(self.encoder_inputs, self.encoder_states)
    
    def fit(self, X, y):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, fileName):
        self.model.save_weights("../save/" + fileName)
        
    def load(self, fileName):
        self.model.load_weights("../save/" + fileName)