  
from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

# load the model from disk
filename = 'model_final.pickle'
save = pickle.load(open(filename, 'rb'))
tokenizer=pickle.load(open('tokenizer_final.pickle','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
        	data = [message]
		seq=tokenizer.texts_to_sequences(data)
		padded=pad_sequences(seq,maxlen=100)
		pred=save.predict(padded)
		pred = np.round(pred)
		if pred>0:
			my_prediction='Toxic'
		else:
			my_prediction='Non-Toxic'
	return render_template('result.html',prediction = my_prediction)
if __name__ == '__main__':
	app.run(debug=True)
