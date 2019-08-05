#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import requests
from scrapy.http import TextResponse
from sklearn.feature_extraction.text import TfidfVectorizer

def get_response(url):
    # function for getting Scrapy response
    page = requests.get(url)
    response = TextResponse(url=page.url,body=page.text,encoding="utf-8")
    return response


def text_to_df(text):
    # takes text as a list an input
    # and conducts TF-IDF vectorization
    # outputs a DataFrame
    tf_idf = TfidfVectorizer()
    tfidf_matrix = tf_idf.fit_transform(text)
    words = tf_idf.get_feature_names()
    data = tfidf_matrix.toarray()
    df = pd.DataFrame(data,columns=words)
    return df

