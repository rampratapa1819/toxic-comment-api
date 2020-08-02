from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# load the model from disk
with open('model.pickle','rb') as handle:
    clf=pickle.load(handle)
with open('countvect.pickle','rb') as handle:
    cv=pickle.load(handle)


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		predictions = clf.predict(vect)
		predictions = np.round(predictions)
		my_prediction=list(prediction)
    		output =[]
    		for i in my_prediction:
        		if i==0:
            			output.append('Not Toxic')
        		else:
            			output.append('Toxic')

	return render_template('result.html',prediction = output)
if __name__ == '__main__':
	app.run(debug=True)
