# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:46:47 2018

@author: Srinivas
"""

from __future__ import print_function,division
from future.utils import iteritems
from builtins import range

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud


#we are using ISO-8859-1 encoding to handle invalid characters from csv file
df = pd.read_csv("C:/Users/Srinivas/Desktop/Practice/AV/SpamDetector/spam.csv",encoding = 'ISO-8859-1')

#Data having 5 columns and out of them 3-5 columns are only null values, so better to remove them

df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)

#renaming the data from v1,v2 to labels and data

df.columns = ['labels','data']

#Create binary labels
df['b_labels'] = df['labels'].map({'ham':0,'spam':1})
Y = df['b_labels'].as_matrix()

count_vectorizer = CountVectorizer(decode_error='ignore') #To ignore any invalid utf-8 characters
X = count_vectorizer.fit_transform(df['data'])

#Split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.33)
#Create the model,train it and print scores
model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("Trainscore",model.score(Xtrain,Ytrain))
print("Test score",model.score(Xtest,Ytest))

# Visualizing the data with WordCloud
def visualize(label):
    words = ''
    for msg in df[df['labels']==label]['data']:
        msg = msg.lower()
        words += msg+' '
    wordcloud = WordCloud(width=600,height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
visualize('spam')
visualize('ham')

# We will see what we are getting wrong
df['predictions'] = model.predict(X)
#Things that should not be spam
sneaky_spam = df[(df['predictions']==0)&(df['b_labels']==1)]['data']
for msg in sneaky_spam:
    print(msg)     
#Things should not be spam 
not_actually_spam = df[(df['predictions']==1)&(df['b_labels']==0)]['data']
for msg in not_actually_spam:
    print(msg)    
    

# Implementing Logistic Regression model
    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model2 = LogisticRegression(solver='liblinear',penalty='l1')
model2.fit(Xtrain,Ytrain)
pred = model2.predict(Xtest)
accuracy_score(Ytest,pred)
    





        
        
        
        
        
        


















