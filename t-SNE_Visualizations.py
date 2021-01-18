# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 19:00:14 2020

@author: Vasileios Baltas
"""

import matplotlib.pyplot as plt
import seaborn as sns
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
from collections import Counter
import pickle

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


gender_specific = ['he','his','her','she','him','man','women','men','woman','spokesman','wife','himself','son','mother','father',
'chairman','daughter','husband','guy','girls','girl','boy','boys','brother','spokeswoman','female','sister','male',
'herself','brothers', 'dad', 'actress', 'mom', 'sons', 'girlfriend', 'daughters', 'lady', 'boyfriend', 'sisters', 'mothers',
'king', 'businessman', 'grandmother', 'grandfather', 'deer', 'ladies', 'uncle', 'males', 'congressman', 'grandson',
'bull', 'queen', 'businessmen', 'wives', 'widow', 'nephew','bride', 'females', 'aunt', 'lesbian',
'chairwoman', 'fathers', 'moms', 'maiden', 'granddaughter', 'lads', 'lion', 'gentleman', 'fraternity',
'bachelor', 'niece', 'bulls', 'husbands', 'prince', 'colt', 'salesman', 'hers', 'dude', 'beard', 'filly', 'princess', 'lesbians',
'councilman', 'actresses', 'gentlemen', 'stepfather', 'monks', 'lad', 'sperm', 'testosterone', 'nephews',
'maid', 'daddy', 'mare', 'fiance', 'fiancee', 'kings', 'dads', 'waitress', 'maternal', 'heroine', 'nieces', 'girlfriends', 'sir',
'stud', 'mistress', 'lions', 'womb', 'grandma', 'maternity', 'estrogen', 'widows',
'gelding', 'diva', 'nuns', 'czar','countrymen','penis', 'bloke', 'nun',
'brides', 'housewife', 'spokesmen', 'suitors', 'menopause', 'monastery', 'motherhood', 'brethren', 'stepmother',
'prostate', 'hostess', 'schoolboy', 'brotherhood', 'fillies', 'stepson', 'congresswoman', 'uncles', 'witch',
'monk', 'viagra', 'paternity', 'suitor', 'sorority', 'macho', 'businesswoman', 'gal', 'statesman', 'schoolgirl',
'fathered', 'goddess', 'hubby', 'stepdaughter', 'blokes', 'dudes', 'strongman', 'uterus', 'grandsons', 'studs', 'mama',
'godfather', 'hens', 'hen', 'mommy', 'boyhood', 'baritone', 'grandmothers',
'grandpa', 'boyfriends', 'feminism', 'countryman', 'stallion', 'heiress', 'queens', 'witches', 'aunts', 'semen', 'fella',
'granddaughters', 'chap', 'widower', 'salesmen', 'convent', 'vagina', 'beau', 'beards', 'handyman', 
'maids', 'gals', 'housewives', 'horsemen', 'obstetrics', 'fatherhood', 'councilwoman', 'princes', 'matriarch', 'colts',
'ma', 'fraternities', 'pa', 'fellas', 'councilmen', 'dowry', 'barbershop', 'fraternal', 'ballerina']




#### import the GloVE embedding

filename = r'glove.6B.300d.txt.word2vec'
GloVe = KeyedVectors.load_word2vec_format(filename, binary=False)

#### importing the ANN and SVM models  ##############

model = load_model('ANN_gender_final')
model_names = load_model('ANN_names_final')

filename1 = 'SVM_gender_final.sav'
filename2 = 'SVM_names_final.sav'


svm = pickle.load(open(filename1, 'rb'))
svm_names = pickle.load(open(filename2, 'rb'))



### importing the data from csv file

corpus =[]         
with open('vocab.csv',newline='',encoding="utf-8") as csvfile:
    for row in csv.reader(csvfile):
        corpus.append(row)
        
        
dictionary = Counter(corpus[0])      #### the available vocabulary obtained from CVs
for i in range(1,len(corpus)):
    dictionary.update(Counter(corpus[i]))        



vocab_to_predict = list()         ### vocabulary for which predictions are going to be made

vocab = list(dictionary.keys())   ### corpus vocabulary

for word in vocab:
    if word in GloVe.vocab.keys():
        vocab_to_predict.append(word)



vecs = np.zeros((len(vocab_to_predict),300))
for i in range(len(vocab_to_predict)):
    vecs[i,:] = GloVe[vocab_to_predict[i]]




#################  COMPARISON OF GENDER MODELS  ###############################




predictions = model.predict(vecs)                  ### predict which words from corpus imply gender
predictions = (np.asarray(predictions)).round() 

predictions_svm = svm.predict(vecs)                ### SVM predictions


vecs1 = np.hstack((vecs,predictions))              ### unify the word vecs with the predicted class
vecs2 = np.hstack((vecs,predictions_svm.reshape((11383,1))))

gend = np.zeros((len(gender_specific),300))     ### creating an array with gender specific words
for i in range(len(gender_specific)):           ### and set their label as '2'
    gend[i,:] = GloVe[gender_specific[i]]

groundtruth_class = np.zeros((len(gend),1))
groundtruth_class[:,:] = 2


gend = np.hstack((gend,groundtruth_class))   ### unify gender-specific word vecs with class label





all_words_ANN = np.vstack((vecs1,gend))        ### ANN Dataframe with 3 classes
all_words_ANN = pd.DataFrame(all_words_ANN)

all_words_SVM = np.vstack((vecs2,gend))        ### SVM Dataframe with 3 classes
all_words_SVM = pd.DataFrame(all_words_SVM)




mappings = {0:'neutral', 1:'gender implying', 2:'ground truth'}
all_words_ANN.iloc[:,300].replace(mappings, inplace=True)
all_words_ANN['LABELS'] = all_words_ANN.iloc[:,300]

all_words_SVM.iloc[:,300].replace(mappings, inplace=True)
all_words_SVM['LABELS'] = all_words_SVM.iloc[:,300]


################  T-SNE VIZ  



from sklearn.manifold import TSNE



y2 = TSNE(n_components=2).fit_transform(all_words_ANN.iloc[:,:300])


plt.figure(figsize=(16,10))
sns.set_style("white")
sns.scatterplot(
    x= y2[:,0], y=y2[:,1],
    hue="LABELS",
    data=all_words_ANN,
    legend="full",
    alpha=0.3
    )



yy = TSNE(n_components=2).fit_transform(all_words_SVM.iloc[:,:300])


plt.figure(figsize=(16,10))
sns.set_style("white")
sns.scatterplot(
    x= yy[:,0], y=yy[:,1],
    hue="LABELS",
    data=all_words_SVM,
    legend="full",
    alpha=0.3
    )






################# VISUALIZATIONS FOR NAME-PREDICTING MODEL  #################



predictions_n = model_names.predict(vecs)                  ### predict which words from corpus imply name
predictions_n = (np.asarray(predictions_n)).round() 

predictions_svm_n = svm_names.predict(vecs)

vecs1_n = np.hstack((vecs,predictions_n))
vecs2_n = np.hstack((vecs,predictions_svm_n.reshape((11383,1))))

nam = np.zeros((len(names),300))                ### creating an array with names
for i in range(len(names)):                     ### and set their label as '2'
    nam[i,:] = GloVe[names[i]]

groundtruth_class_n = np.zeros((len(nam),1))
groundtruth_class_n[:,:] = 2


nam = np.hstack((nam,groundtruth_class_n))   ### unify name word vecs with class label



all_names_ANN = np.vstack((vecs1_n,nam))        ### ANN Dataframe with 3 classes
all_names_ANN = pd.DataFrame(all_names_ANN)




all_names_SVM = np.vstack((vecs2_n,nam))        ### SVM Dataframe with 3 classes
all_names_SVM = pd.DataFrame(all_names_SVM)


mappings = {0:'neutral', 1:'name implying', 2:'ground truth'}
all_names_ANN.iloc[:,300].replace(mappings, inplace=True)
all_names_ANN['LABELS'] = all_names_ANN.iloc[:,300]

all_names_SVM.iloc[:,300].replace(mappings, inplace=True)
all_names_SVM['LABELS'] = all_names_SVM.iloc[:,300]


y3 = TSNE(n_components=2).fit_transform(all_names_ANN.iloc[:,:300])
yy3 = TSNE(n_components=2).fit_transform(all_names_SVM.iloc[:,:300])


plt.figure(figsize=(16,10))
sns.set_style("white")
sns.scatterplot(
    x= y3[:,0], y=y3[:,1],
    hue="LABELS",
    data=all_names_ANN,
    alpha=0.3,
    )


plt.figure(figsize=(16,10))
sns.set_style("white")
sns.scatterplot(
    x= yy3[:,0], y=yy3[:,1],
    hue="LABELS",
    data=all_names_SVM,
    alpha=0.3,
    )



