# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:22:37 2020

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
import matplotlib.pyplot as plt
import gensim
from gensim.corpora.dictionary import Dictionary
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score,make_scorer
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
import pickle

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold





#### importing the GloVe 300 dimensional embeddings

filename2 = r'glove.6B.300d.txt.word2vec'
GloVe = KeyedVectors.load_word2vec_format(filename2, binary=False)


####  defining list of names

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




### in GloVe vocabulary words are sorted according to their frequency
### GloVe embedding includes punctuation and stopwords that we are going to remove as is the case in most NLP apps


punctuation = ['.',')','(','[',']','{','}',';',':','-','--','\t','\n','!','?','_','~','#','&','%','$','@','*','+','/','.', ',','"','<','>','=','\'','|','','/a',"''",'mailto','•','^','..','...','https:','https://','http','https']


stopWords.remove('he')                  ### we make an exception for gender specific words
stopWords.remove('him')  
stopWords.remove('his')
stopWords.remove('himself')
stopWords.remove('she')
stopWords.remove('her')
stopWords.remove('hers')
stopWords.remove('herself')




filtered_glove_vocab = []                          ### filtering the vocabulary of GloVe
                                                   ### 324.134 words in total after filtering
for item in GloVe.vocab.keys():
    if  ( (item.isalpha()) and (len(item)<15) and (len(item)>1)and (item not in stopWords) and (item not in punctuation) ):
        filtered_glove_vocab.append(item)



train_specific = random.sample(names,125)               ### split names in training(125 names)
test_specific = []                                      ### and test set (35 names)
for word in names:
    if word not in train_specific:
      test_specific.append(word)




training_glove = {}                                        ### create a training dictionary with 2k words
for word in filtered_glove_vocab[:2000]:
    if ( (word not in train_specific) and (word not in test_specific) ) :
      training_glove[word] = [GloVe[word],0]               ### GloVe[word] is the embedding vector
      
for word in train_specific:
    training_glove[word] = [GloVe[word],1]                 ### adding training names
    


test_glove = {}                                            ###  test dictionary with 700 words
for word in filtered_glove_vocab[2000:2700]:
    if ( (word not in train_specific) and (word not in test_specific) ) :
      test_glove[word] = [GloVe[word],0]    

for word in test_specific:
    test_glove[word] = [GloVe[word],1]                     ### adding test names



training_set = pd.DataFrame.from_dict(training_glove,orient='index')
test_set = pd.DataFrame.from_dict(test_glove,orient='index')



##############   random permutation for our datasets   #######################


training_set = training_set.reindex(np.random.permutation(training_set.index))
test_set = test_set.reindex(np.random.permutation(test_set.index)) 


#### preparing the training set for ML import

i = 0
X_train = np.zeros((2106,300))       #### number of rows according to training_set.iloc[:,0]
x = training_set.iloc[:,0].values
for vector in x:
    X_train[i,:] = vector
    i += 1
    
i = 0
y_train =  np.zeros((2106,))   
y = training_set.iloc[:,1].values
for target in y:
    y_train[i] = target
    i += 1

#####  preparing the test set for ML import


j = 0
X_test = np.zeros((733,300))        #### number of rows according to test_set.iloc[:,0]
x = test_set.iloc[:,0].values
for vector in x:
    X_test[j,:] = vector
    j += 1
    
j = 0
y_test = np.zeros((733,))
y = test_set.iloc[:,1].values
for target in y:
    y_test[j] = target
    j += 1


##### scaling our predictors with StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



##################       SVM Implementation       ############################




f1 = list()
recall = list()
precision = list()

for i in range(50):
    
      svc = LinearSVC(class_weight='balanced',random_state=i)
      svc.fit(X_train,y_train)
      predictions = svc.predict(X_test)
    
      f1.append(f1_score(y_test,predictions))
      recall.append(recall_score(y_test,predictions))
      precision.append(precision_score(y_test,predictions))
      
 
#####  average scores for 50 runs
     
print('Mean test F1 score is',mean(f1),'+/-',np.std(f1))
print('Mean test recall score is',mean(recall),'+/-',np.std(recall))
print('Mean test precision score is',mean(precision),'+/-',np.std(precision))      
      





#######################     final model    #################################


final_glove = {}                                        ### create a training dictionary 
for word in filtered_glove_vocab[:2000]:
    if word not in names :
      final_glove[word] = [GloVe[word],0]             ### model[word] is the embedding vector



for word in names:
    final_glove[word] = [GloVe[word],1]                 ### adding  gender specific



final_set = pd.DataFrame.from_dict(final_glove,orient='index')

final_set = final_set.reindex(np.random.permutation(final_set.index)) 



i = 0
X_train_final = np.zeros((2141,300))       #### according to final_set.iloc[:,0]
x = final_set.iloc[:,0].values
for vector in x:
    X_train_final[i,:] = vector
    i += 1
    
i = 0
y_train_final =  np.zeros((2141,))   
y = final_set.iloc[:,1].values
for target in y:
    y_train_final[i] = target
    i += 1




svc_final_names = LinearSVC(class_weight='balanced',random_state=1)
svc_final_names.fit(X_train_final,y_train_final)


######   saving the final model

#filename = 'SVM_names_final.sav'
#pickle.dump(svc_final_names, open(filename, 'wb'))
