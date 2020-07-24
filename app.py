  
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
		messag = request.form['message']
        	data = [messag]
		def output(txt_values):
			clean = [ ]
			for i in txt_values:
			    clean.append(i))
			seq=tokenizer.texts_to_sequences(clean)
			padded=pad_sequences(seq,maxlen=100)
			pred=save.predict(padded)
			pred = np.round(pred)
			return (pred, txt_values)
		def message(prediction):
			pred, txt_values = prediction
			toxicity = ['Toxic','Severe_Toxic','Obscene','Threat','Insult','Identity_Hate']
			for i, comment in zip(pred, txt_values):
			    print ('\n')
			    print (comment)
			    if i.any() == 0:
				print ('➤ Comment is Neutral')
			    else:
				for j, k in zip(i, toxicity) :
				    if j == 1:
					print ('➤ Comment is',k)
				    else:
					pass
	return render_template('result.html',prediction = message(output(data)))
if __name__ == '__main__':
	app.run(debug=True)
