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
import operator
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

vaccination.user_location

vaccination[vaccination['user_location'] == 'Moskau']['text']


# controllare la lingua
# controllare l' '’'
# rimuovere cancelletto

#Convert text to lowercase
# Ragionare se da usare
vaccination.text = vaccination.text.str.lower()


allComment = vaccination['text'].to_string(index=False)

#rimozione dei numeri
allComment= re.sub(r'\d+','',allComment)

#Remove whitespaces , trim()
allComment = allComment.strip()
#allComment = re.sub(r'\s+', '', allComment)

#Remove punctuation
allComment = allComment.translate(str.maketrans('','',string.punctuation))

#rimozione \n
allComment = allComment.replace("\n"," ")
allComment = allComment.replace("#"," ")

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
e = nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words += ['im', 'youre', 'thats', 'get', 'cant', 'dont', 'go', 
                   'going', 'gonna', 'like', 'got', 'come', 'look',
                   'see', 'theres', 'say', 'want', 'wanna', 'make', 'yes',
                   'right', '']
tokenized_text = allComment
tokenized_text_without_stopwords = []

for token in tokenized_text:
    if token.lower() not in stop_words:
        tokenized_text_without_stopwords.append(token)

len(tokenized_text_without_stopwords) #155402

def wordListToFreqDict(wordlist):
    today1 = datetime.today()
    print(today1)
    wordfreq = [wordlist.count(p) for p in wordlist]
    today2 = datetime.today()
    print(today2 - today1)
    return dict(list(zip(wordlist,wordfreq)))
#7 minuti

diczionary = wordListToFreqDict(tokenized_text_without_stopwords)

massimoValore = max(diczionary.iteritems(), key=operator.itemgetter(1))[0]

max(diczionary, key=diczionary.get)

diczionary['vaccine']

{k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

A = diczionary
newA = dict(sorted(A.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])

sorted(A, key=A.get, reverse=True)[:10]

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################



#33716 righe
vaccination = pd.read_csv("vaccination_all_tweets.csv")
vaccination.columns

# Ragionare se da usare
vaccination.text = vaccination.text.str.lower()

# Preprocessing

## lowerCase
vaccination.text = vaccination.text.str.lower()

## rimozione dei numeri
vaccination.text = vaccination.text.replace(r'\d+','')

## rimozione spazi bianchi
vaccination.text  = vaccination.text.astype(str).str.strip()

# rimozione punteggiatura
# verificato rimuove anche i cancelletti
vaccination.text=vaccination.text.astype(str).str.translate(str.maketrans('','',string.punctuation))

#rimozione \n
vaccination.text = vaccination.text.replace("\n"," ")

#rimozione apice strano
vaccination.text = vaccination.text.replace("'’'"," ")

# TweetTokenizer
tknzr = TweetTokenizer()
vaccination['text'] = vaccination['text'].apply(tknzr.tokenize)

# Rimozione Stop Words
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words += ['im','youre','thats','get','cant','dont','go',
               'going','gonna','like','got', 'come', 'look','see','theres',
               'say','want','wanna','make','yes','right','']


vaccination['tweet_without_stopwords'] = vaccination['text'].apply(lambda x: [item for item in x if item not in stop_words])
vaccination['tweet_without_stopwords'] 

def wordListToFreqDict(wordlist):
    today1 = datetime.today()
    print(today1)
    wordfreq = [wordlist.count(p) for p in wordlist]
    today2 = datetime.today()
    print(today2 - today1)
    return dict(list(zip(wordlist,wordfreq)))
# 7 minuti

diczionary = wordListToFreqDict(tokenized_text_without_stopwords)

massimoValore = max(diczionary.iteritems(), key=operator.itemgetter(1))[0]

max(diczionary, key=diczionary.get)

A = diczionary
newA = dict(sorted(A.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])

sorted(A, key=A.get, reverse=True)[:10]


# Stemming
# Lemmatization

# sentiment VADER







