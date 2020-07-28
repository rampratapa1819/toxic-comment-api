# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:50:25 2020
@author: Gowrav Tata
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
tokenizer=pickle.load(open('tokenizer.pickle','rb))
model = pickle.load(open('model.pickle', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    tockens=
    prediction = model.predict(final_features)
    prediction=list(prediction)
    output =[]
    for i in prediction:
        if i==0:
            output.append('Not Survived')
        else:
            output.append('Survived')

    return render_template('index.html', prediction_text='Passenger{}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)
