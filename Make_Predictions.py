# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 18:29:52 2020

@author: Vasileios Baltas
"""


import csv
from statistics import stdev,mean
import copy
from random import shuffle
import random
random.seed(33)
import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
np.random.seed(33)
import pandas as pd
import gensim
from gensim.corpora.dictionary import Dictionary
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score,make_scorer
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
import h5py

### some tricks to ensure reproducibility

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONHASHSEED']=str(33)
import tensorflow as tf
tf.compat.v1.set_random_seed(33)

from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score,make_scorer
from tensorflow.keras.models import load_model


names = ['alex','james','kitty','caitlin','roddy','matt','lesley','ross','petra','kulwant',
         'karen','abigail','sue','maryam','denise','julia','paul','robert','dale','victoria','georgia',
         'tom','darren','adam','katy','sarah','tracy','danielle','isis','ravinder','harriet','nicola',
         'leanne','anthony','colette','louise','shannon','sam','soren','micheal','tony','jessica','aaron',
         'stephen','steve','steven','lorna','michelle','shelli','dildar','philip','faheem','kazim',
         'roger','gary','amoako','amok','faisal','ismail','tanya','gem','colin','gemma','donal','rebecca',
         'lisa','lucy','anna','mary','maria','john','george','nick','natasha','natalie','livia','amelia',
         'andrew','anita','alisha','angela','annette','anais','azhar','bernadette','harminder','bruce',
         'rory','carl','carolyn','janet','christopher','charlotte','damian','dan','david','helen','emily',
         'eleonora','elliott','emma','frances','francis','frank','shayne','hashem','iain','ian','irina',
         'jamal','jon','jonathan','julio','junaid','jyoti','roshan','konstantinos','leona','lynn',
         'mark','mohsin','martin','matias','maz','michael','nahid','neil','ni','nicole','niraj','rosie',
         'paul','rachael','rachel','richard','rob','roderick','saq','selwyn','shahid','sharad','shaun',
         'simon','stanley','stephan','usman','leong','wye','ali','harry','anastasia','oliver','chris',
         'alice','joanna','nathan','luis','patrice','margaret','arthur']




#### import the GloVE embedding

filename2 = r'glove.6B.300d.txt.word2vec'
GloVe = KeyedVectors.load_word2vec_format(filename2, binary=False)

#### importing the ANN models

model = load_model('ANN_gender_final')
model_names = load_model('ANN_names_final')


corpus =[]         
with open('vocab.csv',newline='',encoding="utf-8") as csvfile:
    for row in csv.reader(csvfile):
        corpus.append(row)
        
        
banned_words = []
banned_names = []

available_vocab = []                 
for word in corpus[7]:              ### specify which document of the corpus we want to clean
    if word in GloVe.vocab.keys():
        available_vocab.append(word)      ### verifying which words of corpus are available in GloVe
    if word in names:
        banned_names.append(word)

vecs = np.zeros((len(available_vocab),300))
for i in range(len(available_vocab)):
    vecs[i,:] = GloVe[available_vocab[i]]          ### adding GloVe vector to the predictors set
 
predictions = model.predict(vecs)    
predictions = (np.asarray(predictions)).round()    ### predicting gender-implying words
    
predictions = predictions.reshape(len(predictions),)
banned_indexes = np.where(predictions == 1)           ### the index of words to 'blind'


names_predictions = model_names.predict(vecs)                 ### predicting  names
names_predictions = (np.asarray(names_predictions)).round()

names_predictions = names_predictions.reshape(len(names_predictions),)
banned_indexes_names = np.where(names_predictions == 1)




for i in range(len(banned_indexes[0])):
    
    if len(available_vocab[banned_indexes[0][i]])>3 :    ### make sure the output words make sense
      banned_words.append(available_vocab[banned_indexes[0][i]])



for i in range(len(banned_indexes_names[0])):
    
    if len(available_vocab[banned_indexes_names[0][i]])>3 :    ### make sure the output words make sense
        banned_names.append(available_vocab[banned_indexes_names[0][i]])

banned_vocabulary = banned_words + banned_names

print(banned_vocabulary)          #### banned vocabulary are the words predicted to imply gender







words_dic = {}
names_dic = {}

word_pred = [model.predict(GloVe[word].reshape(1,-1)) for word in banned_words]
names_pred = [model_names.predict(GloVe[word].reshape(1,-1)) for word in banned_names] 


for word in banned_words:
    words_dic[word] = model.predict(GloVe[word].reshape(1,-1))           #### words_dic contains the output probabilities of the gender-predicting model

for word in banned_names:
    names_dic[word] = model_names.predict(GloVe[word].reshape(1,-1))     #### names_dic contains the output probabilities of the names-predicting model



