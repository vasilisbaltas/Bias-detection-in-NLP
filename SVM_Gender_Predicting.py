# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:02:22 2020

@author: Vasileios Baltas
"""

from statistics import stdev,mean
import copy
from random import shuffle
import random
import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import pandas as pd
import gensim
from gensim.corpora.dictionary import Dictionary
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
import pickle



filename2 = r'glove.6B.300d.txt.word2vec'
GloVe = KeyedVectors.load_word2vec_format(filename2, binary=False)


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



#### here we define a list of gender specific words as adapted from Bolukbasi et al.(2016)  -  205 words

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





filtered_glove_vocab = []                          ### filtering the vocabulary of GloVe
                                                   ### 324.134 words in total after filtering
for item in GloVe.vocab.keys():
    if  ( (item.isalpha()) and (len(item)<15) and (len(item)>1)and (item not in stopWords) and (item not in punctuation) ):
        filtered_glove_vocab.append(item)
   


train_specific = random.sample(gender_specific,160)     ### split gender specific words in training
test_specific = []                                      ### and test set 
for word in gender_specific:
    if word not in train_specific:
      test_specific.append(word)
    



training_glove = {}                                        ### create a training dictionary with 5k neutral words and their vectors
for word in filtered_glove_vocab[:5145]:
    if ( (word not in train_specific) and (word not in test_specific) ) :
      training_glove[word] = [GloVe[word],0]               ### GloVe[word] is the embedding vector
      
for word in train_specific:
    training_glove[word] = [GloVe[word],1]                 ### adding training gender specific words
    
      

test_glove = {}                                            ###  test dictionary with 700 neutral words
for word in filtered_glove_vocab[5145:5845]:
    if ( (word not in train_specific) and (word not in test_specific) ) :
      test_glove[word] = [GloVe[word],0]    

for word in test_specific:
    test_glove[word] = [GloVe[word],1]                     ### adding test gender specific words



training_set = pd.DataFrame.from_dict(training_glove,orient='index')
test_set = pd.DataFrame.from_dict(test_glove,orient='index')

   
##############   random permutation for our datasets   #######################

np.random.seed(33)
training_set = training_set.reindex(np.random.permutation(training_set.index))
test_set = test_set.reindex(np.random.permutation(test_set.index)) 



#### preparing the training set for ML import

i = 0
X_train = np.zeros((5259,300))       #### number of rows according to training_set.iloc[:,0]
x = training_set.iloc[:,0].values
for vector in x:
    X_train[i,:] = vector
    i += 1
    
i = 0
y_train =  np.zeros((5259,))   
y = training_set.iloc[:,1].values
for target in y:
    y_train[i] = target
    i += 1

#####  preparing the test set for ML import


j = 0
X_test = np.zeros((736,300))         #### number of rows according to test_set.iloc[:,0]
x = test_set.iloc[:,0].values
for vector in x:
    X_test[j,:] = vector
    j += 1
    
j = 0
y_test = np.zeros((736,))
y = test_set.iloc[:,1].values
for target in y:
    y_test[j] = target
    j += 1 


##### scaling our predictors with StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


####################   SVM implementation   ####################################

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score,make_scorer

       


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
                                                      


###########  average scores for 50 runs

print('Mean test F1 score is',mean(f1),'+/-',np.std(f1))
print('Mean test recall score is',mean(recall),'+/-',np.std(recall))
print('Mean test precision score is',mean(precision),'+/-',np.std(precision))
 





########## fit the model to ALL the available gender specific words  ##########


final_glove = {}                                        ### create a training dictionary 
for word in filtered_glove_vocab[:5000]:
    if word not in gender_specific :
      final_glove[word] = [GloVe[word],0]             ### model[word] is the embedding vector



for word in gender_specific:
    final_glove[word] = [GloVe[word],1]                 ### adding  gender specific



final_set = pd.DataFrame.from_dict(final_glove,orient='index')

final_set = final_set.reindex(np.random.permutation(final_set.index)) 



i = 0
X_train_final = np.zeros((5160,300))       #### according to final_set.iloc[:,0]
x = final_set.iloc[:,0].values
for vector in x:
    X_train_final[i,:] = vector
    i += 1
    
i = 0
y_train_final =  np.zeros((5160,))   
y = final_set.iloc[:,1].values
for target in y:
    y_train_final[i] = target
    i += 1




svc_final = LinearSVC(class_weight='balanced',random_state=1)
svc_final.fit(X_train_final,y_train_final)


####  save the final model

#filename = 'SVM_gender_final.sav'
#pickle.dump(svc_final, open(filename, 'wb'))












