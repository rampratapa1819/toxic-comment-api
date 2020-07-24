  
from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from lemmagen3 import Lemmatizer
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
#	df= pd.read_csv("spam.csv", encoding="latin-1")
#	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
#	# Features and Labels
#	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
#	X = df['message']
#	y = df['label']
#	
#	# Extract Feature With CountVectorizer
#	cv = CountVectorizer()
#	X = cv.fit_transform(X) # Fit the Data
#    
#    pickle.dump(cv, open('tranform.pkl', 'wb'))
#    
#    
#	from sklearn.model_selection import train_test_split
#	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#	#Naive Bayes Classifier
#	from sklearn.naive_bayes import MultinomialNB
#
#	clf = MultinomialNB()
#	clf.fit(X_train,y_train)
#	clf.score(X_test,y_test)
#    filename = 'nlp_model.pkl'
#    pickle.dump(clf, open(filename, 'wb'))
    
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
        data = [message]
        ############## Text cleaning
    stopwords_ = ["s","i","me",'my','myself','we','our','ours','a','the','your','ourselves','you',"you're","you've","you'll","you'd",'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', "she","she's", 'her', 'hers', 'herself', 'it', "it's",'its', 'itself', 
    'they', 'them', 'their','theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",'these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as',
    'until','while','of','at','by','for','with','about','against', 'between','into','through','during','before','after','above','below','to',
    'from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all',
    'any','both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
    's','t', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',
    "mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',
    "wouldn't",'best','different',"won't","couldn't","mustn't","didn't","aww","hey","man","women"]

    custom_stop = ["adv", "fundamentals", "course", "advance", "beginner", "beginners","nbsp", "help", "would", "could", "courses","also","get",
    "give","class","much",'copy0','copy1','nbsp','course','copy','best','different']

    stopwords_= stopwords_ + custom_stop
    ## Language processing
    lem = Lemmatizer('en')

    ############## Text cleaning function

    def  clean_text(sent):
        tokens = sent.lower().replace('[^a-z ]',' ').split()
        stemmed = [lem.lemmatize(term) for term in tokens 
                   if term not in stopwords_ and len(term) > 2] 
        res = " ".join(stemmed)
        return res

    ############## Function for Predicting output classes
    def output(txt_values):

        clean = [ ]
        for i in txt_values:
            clean.append(clean_text(i))
        seq=tokenizer.texts_to_sequences(clean)
        padded=pad_sequences(seq,maxlen=100)
        pred=save.predict(padded)
        pred = np.round(pred)
        return (pred, txt_values)




    ############### Printing prediction

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
	app.run(debug=True
