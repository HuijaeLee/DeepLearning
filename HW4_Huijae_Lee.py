#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Install pymagnitude
pip install pymagnitude-lite


# In[32]:


# Get dataset from local file. 
# Since google collab is not working with this library, I used jupyter notebook
from pymagnitude import *
Mag_path = "C:/Users/huija/OneDrive/바탕 화면/2025 Spring_GWU/STAT 6260 Statistical Deep Learning/STAT 6260 Week 7/GoogleNews-vectors-negative300 (1).magnitude"
Words = Magnitude(Mag_path)


# In[33]:


Words.distance("cat","dog")


# In[34]:


# we can see vectors has 3000000 words
print(len(Words))


# In[37]:


# 1.	What is the dimensionality of these word embeddings? Provide an integer answer.
# We can see the dimensionality of these word embddings is 300.
Magni = Words.query("cat")
print(Magni.shape)
# we can directly see the dimension with .dim
print("Embedding dimensionality:", Words.dim)


# In[53]:


# 2.	What are the top-5 most similar words to picnic (not including picnic itself)?
# From github README gives two ways we can get the top-5 similar words. Choose 2 to 6 because the first word will be picnic.

# First way - by key
picnic_sim1 = Words.most_similar("picnic", topn=5)  
# print(picnic_sim1)
# Remove picnic
result1 = picnic_sim1  

# Second way - by vector
picnic_sim2 = vectors.most_similar(vectors.query("picnic"), topn = 6)
result2 = picnic_sim2[1:6]

print("First way:")
print(result1)
print('-'*80)
print("Second way:")
print(result2)

# So our answer will be 
# ('picnics', 0.7400875), ('picnic_lunch', 0.721374), ('Picnic', 0.700534), ('potluck_picnic', 0.6683274), ('picnic_supper', 0.65189123)


# In[54]:


#3.	According to the word embeddings, which of these words is not like the others? 
# ['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette']
which_d_m = Words.doesnt_match(['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette'])
print(which_d_m)
# So, tissue doen't matches.


# In[56]:


# 4.	Solve the following analogy: leg is to jump as X is to throw.
positive_neg = Words.most_similar(positive=["leg","throw"], negative=["jump"], topn=1)
print(positive_neg)
# So we get answer as forearm


# In[ ]:




