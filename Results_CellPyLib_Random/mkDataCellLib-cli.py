#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

import sys
import os

import cellpylib as cpl
from PIL import Image


# In[47]:


# !rm data/ -R


# In[2]:


rgb = np.array([
    [0,0,0],
    [1,1,1],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,1],
    [0,1,1],
    [1,1,0]
])

def render(history,k):
    counts = np.array([-np.sum(history == i) for i in range(k)])
    ridx = np.argsort(counts)
    idx = np.array([ np.where(ridx == i)[0][0] for i in range(k) ])
    
    im = np.zeros((history.shape[1], history.shape[0], 3)).astype(np.uint8)
    for y in range(history.shape[1]):
        for x in range(history.shape[0]):
            im[x,y,:] = np.clip(256*rgb[idx[history[x,history.shape[1]-y-1]]],0,255).astype(np.uint8)
    im = Image.fromarray(im)

    return im


# In[12]:


def render(history,k):
    rgb = np.random.rand(7,3)
    counts = np.array([-np.sum(history == i) for i in range(k)])
    ridx = np.argsort(counts)
    idx = np.array([ np.where(ridx == i)[0][0] for i in range(k) ])
    
    im = np.zeros((history.shape[1], history.shape[0], 3)).astype(np.uint8)
    for y in range(history.shape[1]):
        for x in range(history.shape[0]):
            im[x,y,:] = np.clip(256*rgb[idx[history[x,history.shape[1]-y-1]]],0,255).astype(np.uint8)
    im = Image.fromarray(im)

    return im


# In[10]:


for carule in range(5000):
    directory = "data5k/%.06d" % carule
    if not os.path.exists(directory):
        os.mkdir(directory)


# In[ ]:


for carule in range(5000):
    print(carule)
    k = 2 + np.random.randint(6)
    rn = np.random.randint(int(k**(3*k-2)))
    
    f = open("data5k/%.06d/rule.txt" % carule, "w")
    # k, r, type, rule
    # type: 0 = General, 1 = Totalistic
    f.write("%d 1 1 %d" % (k,rn))
    f.close()
    
    for i in range(50):
        ca = cpl.init_random(256)
        ca = cpl.evolve(ca, timesteps=256, apply_rule=lambda n, c, t: cpl.totalistic_rule(n,k=k,rule=rn), r=1)
        im = render(ca,k)
        im.save("data5k/%.06d/%.06d.png" % (carule, i))


# In[16]:


counts = [ np.sum(ca==i) for i in range(k)]


# In[17]:


counts


# In[ ]:




