from flask import Flask,render_template,url_for,request
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
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


        def  clean_text(sent):
        tokens = sent.lower().replace('[^a-z ]',' ').split()
        stemmed = [term for term in tokens if term not in stopwords_ and len(term) > 2] 
        res = " ".join(stemmed)
        return res
    
        def output(txt_values):
            clean = [ ]
            for i in txt_values:
                clean.append(clean_text(i))
            seq=tokenizer.texts_to_sequences(clean)
            padded=pad_sequences(seq,maxlen=100)
            pred=save.predict(padded)
            pred = np.round(pred)
            return pred

	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
