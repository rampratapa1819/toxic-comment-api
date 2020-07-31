from flask import Flask,render_template,url_for,request
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
save = pickle.load(open(filename, 'rb'))
tokenizer = pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		seq=tokenizer.texts_to_sequences(clean)
    		padded=pad_sequences(seq,maxlen=100)
    		pred=save.predict(padded)
    		pred = list(np.round(pred))
		out=[]
		for i in pred:
			if i==0:
				out.append('Non-Toxic')
			else:
				out.append('Toxic')
    
	return render_template('result.html',prediction = out)



if __name__ == '__main__':
	app.run(debug=True)
