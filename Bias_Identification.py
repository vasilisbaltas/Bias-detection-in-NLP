# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:25:20 2020

@author: Vasileios Baltas
"""


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import seaborn as sns
import math




#### load the word2vec embeddings constructed from the available documents

model = gensim.models.Word2Vec.load(r'C:\Users\Vasileios Baltas\Desktop\SCRIPTS\mymodel')
vectors = model.wv


### she-he,  her-his,  woman-man, female-male, natasha-john

differences = np.zeros((5,300))

differences[0,:] = vectors['she'] - vectors['he']
differences[1,:] = vectors['her'] - vectors['his']
differences[2,:] = vectors['woman'] - vectors['man']
differences[3,:] = vectors['female'] - vectors['male']
differences[4,:] = vectors['natasha'] - vectors['john']

 


pca = PCA(n_components=4)                                    ### 4 principal components explain 99% of the variance
principalComponents = pca.fit_transform(differences)         #### 77.8% of the variance explained by the first 2 PCs


principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4'])


print(pca.explained_variance_ratio_)



###################      GloVe embeddings         ##########################

#### load the pre-trained GloVe embeddings

filename2 = r'glove.6B.300d.txt.word2vec'
model2 = KeyedVectors.load_word2vec_format(filename2, binary=False)
vectors2 = model2.wv

differences2 = np.zeros((5,300))

differences2[0,:] = model2['she'] - model2['he']
differences2[1,:] = model2['her'] - model2['his']
differences2[2,:] = model2['woman'] - model2['man']
differences2[3,:] = model2['female'] - model2['male']
differences2[4,:] = model2['natasha'] - model2['john']



pca2 = PCA(n_components=4)                                   
principalComponents2 = pca2.fit_transform(differences2)
print(pca2.explained_variance_ratio_)                       #### 89.3% of the variance explained by the first 2 PCs







###################  barplots for explained variance  #######################

x = np.arange(4)


ratios = list(pca.explained_variance_ratio_)
fig, ax = plt.subplots()                         #### barplot for home-made embeddings
plt.bar(x,ratios,color='darkblue')
plt.xticks(x, ('PC 1', 'PC2', 'PC3', 'PC4'))
plt.xlabel('Principal Components')
plt.ylabel('Percentage of explained variance')
plt.title('Word2vec PCA')
plt.show()



ratios2 = list(pca2.explained_variance_ratio_)
fig, ax = plt.subplots()                         #### barplot for GloVe embeddings
plt.bar(x,ratios2,color='darkblue')
plt.xticks(x, ('PC 1', 'PC2', 'PC3', 'PC4'))
plt.xlabel('Principal Components')
plt.ylabel('Percentage of explained variance')
plt.title('GloVe PCA')
plt.show()









######################    checking direct bias   ###########################


g = differences[0,:]*principalComponents[0,0] + differences[1,:]*principalComponents[1,0] + \
    differences[2,:]*principalComponents[2,0] + differences[3,:]*principalComponents[3,0] + \
    differences[4,:]*principalComponents[4,0]    


g2 = differences2[0,:]*principalComponents2[0,0] + differences2[1,:]*principalComponents2[1,0] + \
    differences2[2,:]*principalComponents2[2,0] + differences2[3,:]*principalComponents2[3,0] + \
    differences2[4,:]*principalComponents2[4,0] 






vectors.add('g',g)         ### adding the calculated vectors to our embeddings
vectors2.add('g2',g2)


### defining a set of 55 business context neutral words through words from corpus

business_neutral = ['business','management','development','team','experience','manager','project','innovation','skills','support','strategy','service','work',
                  'marketing','company','digital','customer','senior','sales','technology','social','group','delivery','university','responsible','product',
                  'managing','media','growth','recruitment','lead','global','design','financial','client','performance','strategic','scientist','process','data',
                  'commercial','director','research','leadership','head','professional','reporting','analysis','knowledge','system','market',
                  'strong','education','industry','implementation']



def DirectBias(domain_neutral,vectors,g): 
    
  individual = dict()
  N = len(domain_neutral)

  s = 0 
  
  for word in domain_neutral:
    s += abs(vectors.similarity(g,word))
    individual[word] = abs(vectors.similarity(g,word))

  direct_bias = (1/N) * s
  
  return direct_bias,individual

                                                             ### 12.9% for home-made embedding and 17,1% for GloVe
q,individual_1 = DirectBias(business_neutral,vectors,'g')      
print(q)                                                     ### word2vec results
print(individual_1)                                    
                                             
    
q2,individual_2 = DirectBias(business_neutral,vectors2,'gg')
print(q2)                                                    ### GloVe results
print(individual_2)        





######################  checking indirect bias ###############################



def IndirectBias(word,word2,vectors,g):
  
  g = g/(math.sqrt(np.sum(g**2)))                           ### we need to normalize the vectors to be of unit length as proposed by Bolukbasi et al.(2016 - supplement)
  
  vectors[word] = vectors[word]/(math.sqrt(np.sum(vectors[word]**2)))
  vectors[word2] = vectors[word2]/(math.sqrt(np.sum(vectors[word2]**2)))  
    
  word_g = (np.dot(vectors[word],g)) * g
  word_compl = vectors[word] - word_g
  

  word2_g = (np.dot(vectors[word2],g)) * g
  word2_compl = vectors[word2] - word2_g
  

  β =  ( np.dot(vectors[word],vectors[word2]) - ( (np.dot(word_compl,word2_compl))/(np.linalg.norm(word_compl)*np.linalg.norm(word2_compl)) ) ) / (np.dot(vectors[word],vectors[word2])) 

  return β




           
occupations_neutral = ['accountant','administrator','leader','executive','architect','artist','professor',
                       'director','manager','coach','chef','consultant','deputy','entrepreneur','lawyer',
                       'lecturer','officer','researcher','scientist','secretary']


### we are going to check indirect bias between the above gender neutral occupation words
### with the words: dance & footbal

bias_dance_glove = {}
bias_football_glove ={}


bias_dance_domain = {}
bias_football_domain ={}
    
for word in occupations_neutral:
    bias_dance_glove[word] = IndirectBias(word,'dance',vectors2,g2)
    bias_football_glove[word] = IndirectBias(word,'football',vectors2,g2)

    bias_dance_domain[word] = IndirectBias(word,'dance',vectors,g)
    bias_football_domain[word] = IndirectBias(word,'football',vectors,g)



bias_dance_glove = sorted(zip(bias_dance_glove.values(),bias_dance_glove.keys()),reverse=True)
bias_football_glove = sorted(zip(bias_football_glove.values(),bias_football_glove.keys()),reverse=True)
bias_dance_domain = sorted(zip(bias_dance_domain.values(),bias_dance_domain.keys()),reverse=True)
bias_football_domain = sorted(zip(bias_football_domain.values(),bias_football_domain.keys()),reverse=True)


print('Bias associated with dance - GloVe \n', bias_dance_glove)         ### for GloVe embeddings
print('Bias associated with football - GloVe \n', bias_football_glove)
print('Bias associated with dance - word2vec \n',bias_dance_domain)      ### for word2vec embeddings
print('Bias associated with football - word2vec \n',bias_football_domain)


#### positive percentages mean that gender direction enhances the similarity between words
#### while negative percentages imply that gender reduces word similarity
#### we should also take into consideration that gender direction might not be well shaped because of the few word differences ???
#### i.e. in word2vec we see that accountant is affected both from the word dance and football --> 10-15%
#### while GloVE has more solid results i.e. almost all words that are negatively affected regarding dance are possitively affected regarding football
#### check examples @ https://github.com/tolga-b/debiaswe/blob/master/tutorial_example1.ipynb




