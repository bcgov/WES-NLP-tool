# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:04:55 2017

@author: Nasim Taba, Rodrigo Coronado, Pratheek Devaraj
"""

# LIBRARIES
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# PATHS
data_path = "C:/Users/rorui/Desktop/data/JIRA.csv"
survey = pd.read_csv(data_path, encoding='ISO-8859-1')


def azureml_main(dataframe1, _k=10):
    # Load data
    survey_df = dataframe1
    
    # Bag of words. Dictionary
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = 'english', max_features = 200)
    vectorizer.fit(survey_df["Summary"].values.astype('U'))
    
    # Vectorize comments
    X = vectorizer.transform(survey_df["Summary"].values.astype('U'))

    # LDA model
    k = _k
    lda = LDA(learning_method='batch', n_topics=k)
    lda.fit(X)

    # Normalize probabilities
    def normalize(probs):
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]
    
    # Topics with words probability. Matrix with k topics x 200 words
    words_probs = lda.components_

    # Convert vocabulary list to array, so it can be searchable
    dictionary = np.asarray(vectorizer.get_feature_names())

    # Topics words and probabilities in order
    # 2 arrays with col=number of topics
    topics = []
    probabilities = []
    for i in range(0,k):
        words_arranged = dictionary[np.argsort(words_probs[i])]
        topics.append(words_arranged[::-1])
        probs_arranged = np.sort(normalize(words_probs[i])) #normalizing probs so they make sense
        probabilities.append(probs_arranged[::-1])
        
    # Output Data frame
    dataframe1 = pd.DataFrame(index=range(15))
    a=0
    for i in range(k):
        dataframe1[a] = pd.DataFrame(topics[i][0:20])
        dataframe1[a+1] = pd.DataFrame(probabilities[i][0:20])
        a=a+2
    
    return dataframe1

result = azureml_main(survey, 10)

result.to_csv('lda_opendata.csv') 
