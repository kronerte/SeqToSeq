#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:28:05 2019

@author: kronert
"""
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
class SequentialDecoder():
    def __init__(self, num_decoder_tokens, latent_dim, encoder_states):
        self.latent_dim=latent_dim
        # Input
        self.decoder_inputs = Input(shape=(None, num_decoder_tokens))
        # LSTM
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                             initial_state=encoder_states)
        # Dense
        self.decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        
        
        self.model = self.toPredictor()
    
    
    def toPredictor(self):
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = self.decoder_lstm(
             self.decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        return Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [self.decoder_outputs] + decoder_states)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, fileName):
        self.model.save_weights("../save/" + fileName)
        
    def load(self, fileName):
        self.model.load_weights("../save/" + fileName)