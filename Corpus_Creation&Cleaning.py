# -*- coding: utf-8 -*-

import csv
import os
from docx2python import docx2python
import PyPDF2
import pdfplumber
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import nltk
nltk.download('punkt')
nltk.download('stopwords')


nlp = spacy.load('en_core_web_sm')

stopWords = stopwords.words('english')          #### exception of words that we need for bias analysis
stopWords.remove('he')
stopWords.remove('him')  
stopWords.remove('his')
stopWords.remove('himself')
stopWords.remove('she')
stopWords.remove('her')
stopWords.remove('hers')
stopWords.remove('herself')


pattern1 = r'\S+@\w+\.\w+'                     #### define patterns to search for in the CVs
pattern2 = r'\S+com'
pattern3 = r'\S+co.uk'
pattern4 = r'www\S\w+\S+'
pattern5 = r'http:\S\w+\S+'
pattern6 = r'https:.+\s'

pattern_word = r'\S+docx'
pattern_pdf = r'\S+pdf'

punctuation = [')','(','[',']','{','}',';',':','-','--','\t','\n','!','?','_','~','#','&','%','$','@','*','+','/','.', ',','"','<','>','=','\'','|','','/a',"''",'mailto','•','^','..','...','https:','https://','http','https']





def read_text(basepath):
    
  documents_text = []                        ### is going to be list of lists containing every text's keywords
  with os.scandir(basepath) as entries:
    for entry in entries:
        try:
         if entry.is_file():
           if (len(re.findall(pattern_word,entry.name))>0):
               doc = docx2python(basepath+entry.name)   ##docc
               doc_spacy = nlp(doc.text)
               x = analyze_word(doc,doc_spacy)
               documents_text.append(x)
           elif (len(re.findall(pattern_pdf,entry.name))>0):
               text2 = ''
               with pdfplumber.open(basepath+entry.name) as pdf:   
                 pages = pdf.pages
                 for page in pages:
                   text2 += page.extract_text()
               text_spacy = nlp(text2)
               x = analyze_pdf(text2,text_spacy)
               documents_text.append(x)
        except:
            pass
               
               
  return documents_text               


#####  specific function for word files
  
def analyze_word(docc,docc_spacy):
    
    
    p1 = re.findall(pattern1,docc.text)
    p2 = re.findall(pattern2,docc.text)
    p3 = re.findall(pattern3,docc.text)
    p4 = re.findall(pattern4,docc.text)
    p5 = re.findall(pattern5,docc.text)
    p6 = re.findall(pattern6,docc.text)
    pdoc = p1+p2+p3+p4+p5+p6
    pdoc = list(set(pdoc))
    
    
    dates1 = []               #### capturing dates, cities,countries from word - name capturing had problems
    areas1 = []
    for ent in docc_spacy.ents:
      if ent.label_ == 'DATE': 
        dates1.append(ent.text.split())
      elif ent.label_ == 'GPE':
        areas1.append(ent.text)

    d1 =[]                                ### turn dates into separate string i.e. 'July 2019' --> 'July','2019'
    for i in range(len(dates1)):
      for j in range(len(dates1[i])):
        d1.append(dates1[i][j])
          
    d1 = list(set(d1))                   ### get the unique dates
    areas1 = list(set(areas1))           ### get the unique areas


    ### tokenize words

    tokens = word_tokenize(docc.text)


        
    for i,word in enumerate(tokens):                 ### delete GPE and DATES words, gmails and webpages
      if word in d1:
        tokens.remove(word)
      elif word in areas1:
        tokens.remove(word)
      elif word in pdoc:
        tokens.remove(word) 
      elif word == '@':
        tokens.remove(tokens[i-1])                   ### in order to remove baltasvasileios before @


    tokens = [word.lower() for word in tokens]
    key_words = [word for word in tokens if not word in punctuation and len(word)>1 and word.isalpha()]
    keywords = [word for word in key_words if word not in stopWords]
    
    return keywords




#### separate function for pdf files

def analyze_pdf(text22,textt_spacy):
    
  p1 = re.findall(pattern1,text22)
  p2 = re.findall(pattern2,text22)
  p3 = re.findall(pattern3,text22)
  p4 = re.findall(pattern4,text22)
  p5 = re.findall(pattern5,text22)
  p6 = re.findall(pattern6,text22)
  ppdf = p1+p2+p3+p4+p5+p6
  ppdf = list(set(ppdf))

 
  dates2 = []              #### capturing dates, cities,countries from pdf - name capturing had problems
  areas2 = []
  for ent in textt_spacy.ents:
    if ent.label_ == 'DATE':
      dates2.append(ent.text)
    elif ent.label_ == 'GPE':
      areas2.append(ent.text)
        



  d = []                             
  d2 = []                            ### turn dates into separate string i.e. 'July 2019' --> 'July','2019'
  for word in dates2:
    word = word.split()
    d.append(word)
  for i in range(len(d)):               
      for j in range(len(d[i])):
          d2.append(d[i][j])

  d2 = list(set(d2))
  areas2 = list(set(areas2))
  
  
  tokens2 = word_tokenize(text22)
  
  for i,word in enumerate(tokens2):                ### delete GPE and DATES words, gmails and webpages
     if word in d2:
         tokens2.remove(word)
     elif word in areas2:
         tokens2.remove(word)
     elif word in ppdf:
         tokens2.remove(word)
     elif word == '@':
         tokens2.remove(tokens2[i-1]) 
      
        
  tokens2 = [word.lower() for word in tokens2]
  key_words2 = [word2 for word2 in tokens2 if not word2 in punctuation and len(word2)>1 and word2.isalpha()]
  keywords2 = [word2 for word2 in key_words2 if word2 not in stopWords]

  return keywords2       
  
  
  
path = r'C:\\Users\Vasileios Baltas\Desktop\All_files\\'     ##### give the path of the CVs directory !!!!!!
vocabulary = read_text(path)


####  store the cleaned documents ready for analysis in a csv file

with open('vocab.csv','w',newline='',encoding="utf-8") as csvfile:
    for document in vocabulary:
        csv.writer(csvfile).writerow(document)




