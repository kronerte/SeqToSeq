#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:39:16 2019

@author: kronert
"""

#!flask/bin/python
from flask import Flask
from flask import request, jsonify

##########################################################
#from .ML.scr.Translator import Translator
#model = Translator()

class model():
    def __init__(self):
        pass
    def predict(self,L):
        res = []
        for l in L:
            res.append(l+1)
        return res
models_save = []

#model.load(*models_save)
##########################################################

app = Flask(__name__)


@app.route('/predict')
def predict_wrapper():
    content = request.get_json()
    return jsonify(
            model.predict(content["features"])
            )



if __name__ == '__main__':
    app.run(debug=True)