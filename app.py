from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# load the model from disk
filename = 'nlp_model (1).pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('countvect (1).pkl','rb'))


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
		my_predictions = np.round(predictions)

	return render_template('result.html',prediction = my_predictions)
if __name__ == '__main__':
	app.run(debug=True)
