#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:58:52 2019

@author: kronert
"""
import pickle
import numpy as np
class Caracters():
    # Hight level Methods
    def __init__(self):
        self._char_to_int = dict()
        self._int_to_char = []
        self.voc_size = 0
        self.max_sentence_size = 0
        
    def fit(self, L):
        all_char = set()
        m = 0
        for s in L:
            all_char.update(list(s))
            if len(s)>m:
                m = len(s)
        self.max_sentence_size = m
        chars = sorted(list(all_char))
        self.voc_size = len(chars)
        self._int_to_char = chars
        self._char_to_int = dict([(chars[i],i) for i in range(len(chars))])
    
    def predict(self, L, fromSeqToInt = True):        
        if fromSeqToInt:
            n = len(L)
            res = np.zeros((n, self.max_sentence_size, self.voc_size))
            for i in range(n):
                res[i] = self.vectoriseSetence(L[i])
        else:
            n = L.shape[0]
            res = []
            for i in range(n):
                res.append(self.decodeVec(L[i]))
                
        return res
    
    
    def save(self, fileName):
        with open(fileName,"wb") as f:
            pickle.dump(self.__dict__, f)
        
    def load(self, fileName):
        with open(fileName,"rb") as f:
            self.__dict__.update(pickle.load(f))
        
    
    
    # Low lev Methods
    def getChar(self, i):
        return self._int_to_char[i]
    
    def getInt(self, c):
        return self._char_to_int[c]
    
    def vectoriseSetence(self, sentence):
        res = np.zeros((self.max_sentence_size, self.voc_size))
        for i in range(len(sentence)):
            res[i][self.getInt(sentence[i])] = 1
        return res
    
    def decodeVec(self, vec):
        res = ""
        for i in range(vec.shape[0]):
            res += self.getChar(np.argmax(vec[i]))
        return res
    