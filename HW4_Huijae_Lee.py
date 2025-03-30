


#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Install pymagnitude
pip install pymagnitude-lite


# In[2]:


# Get dataset from local file. 
# Since google collab is not working with this library, I used jupyter notebook
from pymagnitude import *
Mag_path = "C:/Users/huija/OneDrive/바탕 화면/2025 Spring_GWU/STAT 6260 Statistical Deep Learning/STAT 6260 Week 7/GoogleNews-vectors-negative300 (1).magnitude"
Words = Magnitude(Mag_path)


# In[3]:


Words.distance("cat","dog")


# In[4]:


# we can see vectors has 3000000 words
print(len(Words))


# In[5]:


# 1.	What is the dimensionality of these word embeddings? Provide an integer answer.
# We can see the dimensionality of these word embddings is 300.
Magni = Words.query("cat")
print(Magni.shape)
# we can directly see the dimension with .dim
print("The dimension will be:", Words.dim)


# In[6]:


# 2.	What are the top-5 most similar words to picnic (not including picnic itself)?
# From github README gives two ways we can get the top-5 similar words. Choose 2 to 6 because the first word will be picnic.

# First way - by key
picnic_sim1 = Words.most_similar("picnic", topn=5)  
# print(picnic_sim1)
# Remove picnic
result1 = picnic_sim1  

# Second way - by vector
picnic_sim2 = Words.most_similar(Words.query("picnic"), topn = 6)
result2 = picnic_sim2[1:6]

print("First way:")
print(result1)
print('-'*80)
print("Second way:")
print(result2)

# So our answer will be 
# ('picnics', 0.7400875), ('picnic_lunch', 0.721374), ('Picnic', 0.700534), ('potluck_picnic', 0.6683274), ('picnic_supper', 0.65189123)


# In[7]:


#3.	According to the word embeddings, which of these words is not like the others? 
# ['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette']
which_d_m = Words.doesnt_match(['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette'])
print(which_d_m)
# So, tissue doen't matches.


# In[8]:


# 4.	Solve the following analogy: leg is to jump as X is to throw.
positive_neg = Words.most_similar(positive=["leg","throw"], negative=["jump"], topn=1)
print(positive_neg)
# So we get answer as forearm


# In[17]:


print(Words.query("leg")[:2])
print(Words.query("jump")[:2])
print(Words.query("throw")[:2])
print(Words.query("forearm")[:2])


# In[28]:


import matplotlib.pyplot as plt

# stock vecctors in word_vec
word_vec = {
    "leg": [0.055176, 0.052734],
    "jump": [0.073242, 0.03418],
    "throw": [-0.067383, 0.032227],
    "forearm": [0.03418, 0.06836]
}

# plot in 2D
plt.figure(figsize=(10, 8))
plt.title("300D to 2D plot")
plt.grid(True)

for word, (x, y) in word_vec.items():
    plt.scatter(x, y)
    plt.text(x + 0.001, y + 0.001, word, fontsize=10)
# This acually does not show the relationship directly because we used 300D to calcualte word realtionships not 2D.
plt.show()


# In[24]:


picnic_sim1 = Words.most_similar("picnic", topn=10)  
print(picnic_sim1)
# print(picnic_sim1)
print(Words.query("picnics")[:2]) 
print(Words.query("picnic_lunch")[:2])
print(Words.query("Picnic")[:2])
print(Words.query("potluck_picnic")[:2])
print(Words.query("picnic_supper")[:2])
print(Words.query("picnicking")[:2])
print(Words.query("cookout")[:2])
print(Words.query("Hiking_biking_camping")[:2])
print(Words.query("barbeque")[:2])
print(Words.query("barbecue")[:2])

print(Words.query("picnic")[:2])


# In[30]:


word_vec2 = {
    "picnic" : [ 0.0583585, -0.0391094],
    "picnics": [ 0.0403712, -0.0058524],
    "picnic_lunch": [-0.0326034, -0.0450561],
    "Picnic": [ 0.0214468, -0.1357783],
    "potluck_picnic": [-0.0116013, -0.055853 ],
    "picnic_supper": [-0.0453479, -0.059007 ],
    "picnicking": [0.0823966, 0.005247 ],
    "cookout": [-0.0078559, -0.0232968],
    "Hiking_biking_camping": [-0.0216939, -0.0320538],
    "barbeque": [-0.0970738,  0.0221316],
    "barbecue": [-0.0548297, -0.0297382]
}

# Plotting
plt.figure(figsize=(10, 8))
plt.title("2D top 10 words - picnic plot")
plt.grid(True)

for word, (x, y) in word_vec2.items():
    plt.scatter(x, y)
    plt.text(x + 0.001, y + 0.001, word, fontsize=10)

plt.show()


# In[ ]:




