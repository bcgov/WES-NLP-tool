# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 00:45:28 2017

@author: Rodrigo Coronado, Nasim Taba, Pratheek Devaraj
This script will take comments and use Cognitive Services for Sentiment Analysis. A Cognitive Services Account needs to 
be setup to make use of the NLP Sentiment service
"""

import pandas as pd
import http.client, urllib.request, urllib.parse, urllib.error, json

# Input and output DataFrames.
survey_df = pd.read_csv("C:/Users/rorui/Desktop/data/combined.csv", encoding='ISO-8859-1')
dataframe1 = survey_df[['ID.2', 'AQ3345_13_CLEAN']]
dataframeR = pd.DataFrame(columns=('id', 'comment', 'sentiment'))

# Prepare Azure Cognitive Services Connection.
headers = headers = {'Content-Type': 'application/json','Ocp-Apim-Subscription-Key': 'c5ab2e7f642641d588f1c6e6bda875b5',}
data = {"documents":[{"id": "0","text": "first document"}]}
params = urllib.parse.urlencode({})
conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')

# Get sentiment analysis.
for i in range(len(dataframe1)):
    # Prep json
    data['documents'][0]['id'] = str(dataframe1['ID.2'][i])
    if(pd.isnull(dataframe1['AQ3345_13_CLEAN'][i])):
        data['documents'][0]['text'] = 'NaNd'
    else:
        data['documents'][0]['text'] = dataframe1['AQ3345_13_CLEAN'][i]
    body = str.encode(json.dumps(data))
    
    # Send json for sentiment evaluation
    conn.request("POST", "/text/analytics/v2.0/sentiment?%s" % params, body, headers)
    response = conn.getresponse()
    result = response.read()
    resultr = json.loads(result.decode('utf-8'))
        
    # Prep dataframe with results
    df_temp = pd.DataFrame({'id':data['documents'][0]['id'],
                            'comment':data['documents'][0]['text'],
                            'sentiment':resultr['documents'][0]['score']}, index=[0])
    dataframeR = dataframeR.append(df_temp)

conn.close()
####################################