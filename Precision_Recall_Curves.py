# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:33:55 2020

@author: Vasileios Baltas
"""


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




#### importing the GloVe 300 dimensional embeddings

filename2 = r'glove.6B.300d.txt.word2vec'
GLoVe = KeyedVectors.load_word2vec_format(filename2, binary=False)


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





################### GENDER PREDICTING MODEL CURVES  #########################





filtered_glove_vocab = []                          ### filtering the vocabulary of GloVe
                                                   ### 324.134 words in total after filtering
for item in GLoVe.vocab.keys():
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
      training_glove[word] = [GLoVe[word],0]               ### GLoVe[word] is the embedding vector
      
for word in train_specific:
    training_glove[word] = [GLoVe[word],1]                 ### adding training gender specific words
    
      

test_glove = {}                                            ###  test dictionary with 700 neutral words
for word in filtered_glove_vocab[5145:5845]:
    if ( (word not in train_specific) and (word not in test_specific) ) :
      test_glove[word] = [GLoVe[word],0]    

for word in test_specific:
    test_glove[word] = [GLoVe[word],1]                     ### adding test gender specific words



training_set = pd.DataFrame.from_dict(training_glove,orient='index')
test_set = pd.DataFrame.from_dict(test_glove,orient='index')

   
##############   random permutation for our datasets   #######################


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




#### creating a Neural Network




from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score,make_scorer
import matplotlib.pyplot as plt


### custom f1-score function

def custom_f1(ytrue,ypred) :
    
    def recall_m(ytrue,ypred):
        TP = K.sum(K.round(K.clip(ytrue * ypred, 0, 1)))
        Positives = K.sum(K.round(K.clip(ytrue, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        
        return recall

    
    def precision_m(ytrue,ypred):
        TP = K.sum(K.round(K.clip(ytrue * ypred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(ypred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        
        return precision
        


    precision,recall = precision_m(ytrue,ypred), recall_m(ytrue,ypred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


from sklearn.svm import SVC, LinearSVC
import sklearn.svm as svm
from sklearn.calibration import CalibratedClassifierCV

svm = LinearSVC(class_weight='balanced',random_state=63)
svc = CalibratedClassifierCV(svm)
svc.fit(X_train,y_train)

q = svc.predict_proba(X_test)
lr_probs = q[:,1]



model3 = Sequential()
model3.add(Dense(600,activation='relu',input_shape=(300,)))
model3.add(Dense(200,activation='relu'))
model3.add(Dropout(0.25))
model3.add(Dense(100,activation = 'relu'))
model3.add(Dense(1,activation='sigmoid'))

           
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=[custom_f1,'accuracy'])
history = model3.fit(X_train,y_train,epochs=6,validation_split=0.1)

y_val_pred = model3.predict(X_test)
predictions = (np.asarray(y_val_pred)).round()


##############  precision-recall curves  #######################


from sklearn.metrics import roc_curve,auc,f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from keras.wrappers.scikit_learn import KerasClassifier



#### ANN details
lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_val_pred)
lr_f1, lr_auc = f1_score(y_test, predictions), auc(lr_recall, lr_precision)

#### SVM details

svm_precision,svm_recall, _ = precision_recall_curve(y_test, lr_probs)
svm_auc = auc(svm_recall, svm_precision)

no_skill = len(y_test[y_test==1]) / len(y_test)

plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision,  label='ANN',color='darkblue')
plt.plot(svm_recall,svm_precision, label='SVM',color='orange')

plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
plt.title('Precision-Recall tradeoff')
plt.show()

print(lr_auc)     ### areas under the curve
print(svm_auc)   












######################   NAME-PREDICTING MODEL CURVES ########################


filtered_glove_vocab = []                          ### filtering the vocabulary of GloVe
                                                   ### 324.134 words in total after filtering
for item in GLoVe.vocab.keys():
    if  ( (item.isalpha()) and (len(item)<15) and (len(item)>1)and (item not in stopWords) and (item not in punctuation) ):
        filtered_glove_vocab.append(item)



train_specific = random.sample(names,125)               ### split names in training(125 names)
test_specific = []                                      ### and test set (35 names)
for word in names:
    if word not in train_specific:
      test_specific.append(word)




training_glove = {}                                        ### create a training dictionary 
for word in filtered_glove_vocab[:2000]:
    if ( (word not in train_specific) and (word not in test_specific) ) :
      training_glove[word] = [GLoVe[word],0]               ### GLoVe[word] is the embedding vector
      
for word in train_specific:
    training_glove[word] = [GLoVe[word],1]                 ### adding training names
    


test_glove = {}                                            ###  test dictionary 
for word in filtered_glove_vocab[2000:2700]:
    if ( (word not in train_specific) and (word not in test_specific) ) :
      test_glove[word] = [GLoVe[word],0]    

for word in test_specific:
    test_glove[word] = [GLoVe[word],1]                     ### adding test names



training_set = pd.DataFrame.from_dict(training_glove,orient='index')
test_set = pd.DataFrame.from_dict(test_glove,orient='index')



##############   random permutation for our datasets   #######################


training_set = training_set.reindex(np.random.permutation(training_set.index))
test_set = test_set.reindex(np.random.permutation(test_set.index)) 



#### preparing the training set for ML import

i = 0
X_train = np.zeros((2107,300))       #### number of rows according to training_set.iloc[:,0]
x = training_set.iloc[:,0].values
for vector in x:
    X_train[i,:] = vector
    i += 1
    
i = 0
y_train =  np.zeros((2107,))   
y = training_set.iloc[:,1].values
for target in y:
    y_train[i] = target
    i += 1

#####  preparing the test set for ML import


j = 0
X_test = np.zeros((733,300))    #### number of rows according to test_set.iloc[:,0]
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




model_n = Sequential()
model_n.add(Dense(600,activation='relu',input_shape=(300,)))
model_n.add(Dense(200,activation='relu'))
model_n.add(Dropout(0.25))
model_n.add(Dense(100,activation = 'relu'))
model_n.add(Dense(1,activation='sigmoid'))

           
model_n.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model_n.fit(X_train,y_train,epochs=4,validation_split=0.2)
#plot_loss(history.history['custom_f1'], history.history['val_custom_f1'])

y_val_pred = model_n.predict(X_test)
predictions = (np.asarray(y_val_pred)).round()




svm_n = LinearSVC(class_weight='balanced',random_state=63)
svc_n = CalibratedClassifierCV(svm_n)
svc_n.fit(X_train,y_train)

q_n = svc_n.predict_proba(X_test)
lr_probs_n = q_n[:,1]



lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_val_pred)
lr_f1, lr_auc = f1_score(y_test, predictions), auc(lr_recall, lr_precision)

#### SVM details

svm_precision,svm_recall, _ = precision_recall_curve(y_test, lr_probs_n)
svm_auc = auc(svm_recall, svm_precision)

no_skill = len(y_test[y_test==1]) / len(y_test)

plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision,  label='ANN',color='darkblue')
plt.plot(svm_recall,svm_precision, label='SVM',color='orange')

plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
plt.title('Precision-Recall tradeoff')
plt.show()

print(lr_auc)    ### areas under the curve
print(svm_auc)   




















