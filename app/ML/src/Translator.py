#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:13:43 2019

@author: kronert
"""

class Translator():
    def __init__(self, languageInput, languageOutput, sequentialEncoderDecoder):
        self.languageInput = languageInput
        self.languageOutput = languageOutput
        self.sequentialEncoderDecoder = sequentialEncoderDecoder
        
    def translate(self, sentance):
        vectorized = self.languageInput.predict([sentance])