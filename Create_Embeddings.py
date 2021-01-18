# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 00:44:07 2020

@author: Vasileios Baltas
"""

from statistics import stdev
import numpy as np
import csv
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
import gensim
import pickle
import logging
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

#### import the processed documents in a list of lists  --> corpus

corpus =[]         
with open('vocab.csv',newline='',encoding="utf-8") as csvfile:
    for row in csv.reader(csvfile):
        corpus.append(row)
        
        
        
#####################  creating the word2vec model   #########################        
        
        
        
##### instantiate the word2vec model

model = gensim.models.Word2Vec(corpus,sg=1,seed=123,workers=1, min_count=3,size=300)     ### sg=1 means skip-gram model
                                                                                         ### we define seed and workers in order to ensure reproducible training

##### train the model on the available corpus

model.train(corpus,total_examples = model.corpus_count,epochs=model.iter)      


##### keep only the produced word vectors

vectors = model.wv
# del model    in order to use less memory

model.wv.similarity('woman','business')   #### an example of similarity


##### save word2vec model

model.save(r'C:\Users\Vasileios Baltas\Desktop\SCRIPTS\mymodel')     ### save the created word2vec model in your directory




################  instantiate the pre-trained GloVe model  ###################


##### import the GloVe embeddings - need to download them from Stanford university website

glove_input_file = r'glove.6B.300d.txt'        ### no need to run this if we already have in our directory the transformed .txt.word2vec model/file


##### transform the GloVe embeddings into word2vec embeddings and save them

word2vec_output_file = r'glove.6B.300d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)


########  loading the GloVe word embeddings    #########

#filename2 = r'glove.6B.300d.txt.word2vec'
#model = KeyedVectors.load_word2vec_format(filename2, binary=False)






