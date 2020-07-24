#### Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from lemmagen3 import Lemmatizer
import re
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import Sequential
import pickle

####  Data Preprocessing
## Stop words
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

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
lem=Lemmatizer('en')
def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    #text = text.replace('x', ' ')
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+',' ',text)
    text=lem.lemmatize(text)
    text = ' '.join(word for word in text.split() if word not in stopwords_) # remove stopwors from text
    return text

#### Data Loading
train=pd.read_csv("/content/drive/My Drive/Files/comments/train.csv")

## Applying Cleaning Function on Data
train['cleaned_text']=train.comment_text.apply(clean_text)

## Preparing Data for Modeling
labels=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
x=train['cleaned_text'].values
y=train[labels].values

## Tokenization using Tokenizer
max_features = 10000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x))
tokenized_train = tokenizer.texts_to_sequences(x)

## Saving Tokenizer as pickle file
with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

## Padding
maxlen = 100
x_train = pad_sequences(tokenized_train, maxlen=maxlen)

## Defining Model
inp = Input(shape=(maxlen, ))
embed_size = 10
x = Embedding(max_features, embed_size)(inp)
x = LSTM(15, return_sequences=True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.2)(x)
x = Dense(6, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(6, activation="sigmoid")(x)
model=Sequential()
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
## checkpoint
filepath='wts.h5'
checkpoint=ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

## Training Model
model.fit(x_train, y, batch_size=30, epochs=3, validation_split=0.33, verbose=1, callbacks=[checkpoint])
with open('model.pickle','wb') as handle1:
    pickle.dump(model,handle1,protocol=pickle.HIGHEST_PROTOCOL)
