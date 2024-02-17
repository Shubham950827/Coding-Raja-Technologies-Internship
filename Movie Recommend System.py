#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv("movies.csv")
credit = pd.read_csv("credits.csv")


# In[3]:


movies


# In[4]:


credit


# In[5]:


movies = movies.merge(credit,on = "title")


# In[6]:


movies


# In[7]:


movies = movies[['movie_id','genres','overview','keywords','title','cast','crew']]


# In[8]:


movies.head()


# In[9]:


movies.info()


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[13]:


import ast


# In[14]:


def convert(obj):
    L =[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[15]:


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[16]:


def characterLst(obj):
    lst = []
    for i in eval(obj)[:3]:
        lst.append(i['character'])
    return lst
            


# In[17]:


movies['cast'] = movies['cast'].apply(characterLst)


# In[18]:


movies


# In[19]:


def director(obj):
    L =[]
    for i in ast.literal_eval(obj):
        if i['job'] =='Director':
             L.append(i['name'])
    return L
    


# In[20]:


movies['crew'] = movies['crew'].apply(director)


# In[21]:


movies


# In[22]:


movies.head()


# In[23]:


movies['overview'][0]


# In[24]:


movies['overview']= movies['overview'].apply(lambda x:x.split())


# In[25]:


movies


# In[26]:


movies['overview']= movies['overview'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']= movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']= movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']= movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[27]:


movies


# In[28]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[29]:


movies


# In[30]:


df = movies[['movie_id','title','tags']]


# In[31]:


df


# In[32]:


df['tags'] = df['tags'].apply(lambda x:' '.join(x))


# In[33]:


df


# In[34]:


df['tags'] = df['tags'].apply(lambda X:X.lower())


# In[35]:


df


# In[36]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=500,stop_words='english')


# In[37]:


cv.fit_transform(df['tags']).toarray().shape


# In[38]:


vectors = cv.fit_transform(df['tags']).toarray()


# In[39]:


vectors[0]


# In[40]:


len(cv.get_feature_names())


# In[41]:


import nltk


# In[42]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[43]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[44]:


df['tags'] = df['tags'].apply(stem)


# In[45]:


from sklearn.metrics.pairwise import cosine_similarity


# In[46]:


cosine_similarity(vectors)


# In[47]:


cosine_similarity(vectors).shape


# In[48]:


similarity = cosine_similarity(vectors)


# In[49]:


sorted(list(enumerate(similarity[0])), reverse = True, key =lambda x:x[1])[1:6]


# In[58]:


def recommend(movie):
    movie_index = df[df['title'] == movie].index[0]
    dis = similarity[movie_index]
    max_dis = sorted(dis,reverse=True)
    recommend_movies_index = [dis.tolist().index(val) for val in max_dis[:5]]
    names = []
    for index in recommend_movies_index:
        names.append(df.iloc[index].title)
    return names


# In[59]:


recommend('Avatar')


# In[ ]:




