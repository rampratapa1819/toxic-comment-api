####### Imports
from memory_profiler import memory_usage
start_mem=memory_usage()[0]
start_mem1 = memory_usage()[0]

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from lemmagen3 import Lemmatizer
from keras.preprocessing.sequence import pad_sequences
import pickle
end_mem1 = memory_usage()[0]
print ('Memory Usage Imports :', end_mem1 - start_mem1 , 'MB')

start_mem2 = memory_usage()[0]
######### Loading model
with open('model.pickle','rb') as handle:
    save=pickle.load(handle)


end_mem2 = memory_usage()[0]
print ('Memory Usage Model loading :', end_mem2 - start_mem2 , 'MB')

start_mem3 = memory_usage()[0]

######### loading tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)
end_mem3 = memory_usage()[0]
print ('Memory Usage tokenizer  :', end_mem3 - start_mem3 , 'MB')


start_mem4 = memory_usage()[0]
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
###### Preparing Data for Predictions

test_comments = ["Never trust Udemy. It is the most pathetic, unworthy and untrustful site.","When seeing all over the experience with Udemy is too bad and I never recommend anyone to go for udemy. Type of YouTube channels are better than udemy.. At least you can save your money.","Many of my coworkers choose to use Udemy for continuing education. I feel it has the best selection, training and curriculum vs others I have tried. Yes, the courses may be longer than others, but they're more detailed.","This bar sucks plain and simple. Dominated by hipster retro people that can not be talked to unless you know know Morrissey's new album. Pabst always a great price shit bartenders who will ignore you and make drinks like they know what they are doing. This place is great for any 20 something year old trying to fit in haha garbage place. Oh do not forget the sexist 5 dollar make charge lol get fucked","An extremely helpful and informative course, especially in conjuction with multi-modal training. Training materials were well organized and provided good case studies. Instructor was extremely professional and pleasant to learn from. Dawn did an exceptional job presenting the material. She set up by explaining what she was going to teach us, summarized, and proceeded to teach, providing relevant real life examples. She found out what we handled and catered examples to us to make the course meaningful. I am a CHMM and have taken many similar courses - this was very well done, which I attribute primarily to the instructor and secondarily to the quality materials."]


#### Final
message(output(test_comments))
end_mem4 = memory_usage()[0]
end_mem=memory_usage()[0]
print ('Memory Usage predictions:', end_mem4 - start_mem4 , 'MB')
print('Total Memory usage:',end_mem-start_mem,'MB')
