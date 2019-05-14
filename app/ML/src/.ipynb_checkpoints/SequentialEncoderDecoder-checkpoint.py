#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:59:01 2019

@author: kronert
"""

import numpy as np
from .SequentialEncoder import SequentialEncoder
from .SequentialDecoder import SequentialDecoder

class SequentialEncoderDecoder():
    def __init__(self,num_encoder_tokens, latent_dim_enc, 
                 num_decoder_tokens, latent_dim_dec):
        #Encoder
        self.encoder = SequentialEncoder(num_encoder_tokens, latent_dim_enc)
        #Decoder
        self.decoder = SequentialDecoder(num_decoder_tokens, latent_dim_dec, self.encoder.encoder_states)
        # Model for training
        self.trainModel =Model([self.encoder.encoder_inputs, self.decoder.decoder_inputs],
                      self.decoder.decoder_outputs) 
        self.trainModel.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        
    def fit(self, encoder_input_data, decoder_input_data, decoder_target_data,
            batch_size=64, epochs=100,  validation_split=0.2):
        self.trainModel.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=validation_split)
    
    def predict(self, sentences):
       return None
        
    def save(self, fileName):
        self.encoder.save('enc_' + fileName)
        self.decoder.save('dec_' + fileName)
        
    def load(self, fileName):
        self.encoder.load('enc_' + fileName) 
        self.decoder.load('dec_' + fileName) 