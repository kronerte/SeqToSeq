#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:22:52 2019

@author: kronert
"""


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
# Defining some constants: 
vec_len       = 300   # Length of the vector that we willl get from the embedding layer
latent_dim    = 1024  # Hidden layers dimension 
dropout_rate  = 0.2   # Rate of the dropout layers
batch_size    = 64    # Batch size
epochs        = 30    # Number of epochs
num_en_words = 32

# Define an input sequence and process it.
# Input layer of the encoder :
encoder_input = Input(shape=(None,50))

# Hidden layers of the encoder :
encoder_embedding = Embedding(input_dim = num_en_words, output_dim = vec_len)(encoder_input)
encoder_dropout   = (TimeDistributed(Dropout(rate = dropout_rate)))(encoder_embedding)
encoder_LSTM      = LSTM(latent_dim, return_sequences=True)(encoder_dropout)

# Output layer of the encoder :
encoder_LSTM2_layer = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_LSTM2_layer(encoder_LSTM)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]