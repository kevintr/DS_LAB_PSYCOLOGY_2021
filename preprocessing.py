# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 21:57:45 2021

@author: TranchinaKe
"""

import os 
import pandas as pd
from datetime import datetime, timedelta
import nltk
import re ##regulare expression
import string
from nltk.tokenize import BlanklineTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import MWETokenizer
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import subprocess
#pip install bcrypt

# per aprire la cartella della mail da cui poi copiare il file
subprocess.Popen(r'explorer /select,"C:\Users\tranchinake\Desktop\DataScienceUniversity\LabBusinessAnalytic"')

#Convert text to lowercase
#Remove numbers
#Remove punctuation
#Tokenization
#Remove stop words
#Stemming
#Lemmatization

path = str(r'C:\Users\tranchinake\Desktop\DataScienceUniversity\LabBusinessAnalytic')
path = path.replace("\\","/")
os.chdir(path)


#33716 righe
vaccination = pd.read_csv("vaccination_all_tweets.csv")
vaccination.columns

#Convert text to lowercase
vaccination.text = vaccination.text.str.lower()

allComment = vaccination['text'].to_string(index=False)

#rimozione dei numeri
allComment= re.sub(r'\d+','',allComment)

#Remove whitespaces
allComment = allComment.strip()
#allComment = re.sub(r'\s+', '', allComment)

#Remove punctuation
allComment = allComment.translate(str.maketrans('','',string.punctuation))

#rimozione \n
allComment = allComment.replace("\n","")

################tokenizazione
##BlanklineTokenizer tokenization
#allComment = BlanklineTokenizer().tokenize(allComment)
#
##WordPunctTokenizer
#allComment = WordPunctTokenizer().tokenize(allComment)
#
##Regexp
#tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
#allComment = tokenizer.tokenize(allComment)

#TweetTokenizer
tknzr = TweetTokenizer()
allComment = tknzr.tokenize(allComment)

len(allComment) # 225578

##MWETokenizer
#tokenizer = MWETokenizer()
#tokenizer.add_mwe(('in', 'spite', 'of'))
#tokenized_text = WordPunctTokenizer().tokenize(allComment)
#allComment = tokenizer.tokenize(allComment)

#Remove stop words
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
tokenized_text = allComment
tokenized_text_without_stopwords = []
for token in tokenized_text:
    if token.lower() not in stop_words:
        tokenized_text_without_stopwords.append(token)

len(tokenized_text_without_stopwords) #155402

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

wordListToFreqDict(tokenized_text_without_stopwords)

#Stemming
#Lemmatization

#Part of speech tagging (POS)

