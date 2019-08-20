#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

import sys
import os

from PIL import Image


# In[2]:


for carule in range(2400):
    os.makedirs("data/%.06d" % (carule))


# In[8]:


class CA():
    def __init__(self, colors = 3, XR=128, TR = 128):
        self.rgb = np.random.rand(colors,3)
        self.state = np.random.randint(colors, size=(XR,)).astype(np.int32)
        
        self.history = np.zeros((XR,TR)).astype(np.int32)
        self.history[:,0] = self.state
        
        self.colors = colors
        self.transitions = np.random.randint(colors, size=(4**colors))
        
    def runca(self):
        for t in range(self.history.shape[1]):
            self.history[:,t] = self.state
            neighborhoods = np.array([np.roll(self.state,-1,0), self.state, np.roll(self.state,1,0)])
            idx = neighborhoods[0,:] + self.colors * neighborhoods[1,:] + self.colors * self.colors * neighborhoods[2,:]
            self.state = self.transitions[idx]
            
    def render(self):
        im = np.zeros((self.history.shape[1], self.history.shape[0], 3)).astype(np.uint8)
        for y in range(self.history.shape[1]):
            for x in range(self.history.shape[0]):
                im[y,x,:] = np.clip(256*self.rgb[self.history[x,y]],0,255).astype(np.uint8)
        im = Image.fromarray(im)
        
        return im


# In[9]:


for carule in range(600):
    if carule<300:
        ca = CA(colors = 3)
    else:
        ca = CA(colors = 4)
        
    for example in range(50):
        ca.rgb = np.random.rand(4,3)
        ca.state = np.random.randint(4, size=(128,))
        ca.runca()
        
        im = ca.render()
        im.save("data/%.06d/%.06d.png" % (carule, example))


# In[4]:


class TotalCA():
    def __init__(self, colors = 3, XR=128, TR = 128):
        self.rgb = np.random.rand(colors,3)
        self.state = np.random.randint(colors, size=(XR,)).astype(np.int32)
        
        self.history = np.zeros((XR,TR)).astype(np.int32)
        self.history[:,0] = self.state
        
        self.colors = colors
        self.transitions = np.random.randint(colors, size=(6**colors))
        
    def runca(self):
        for t in range(self.history.shape[1]):
            self.history[:,t] = self.state
            neighborhoods = np.array([np.roll(self.state,-1,0), self.state, np.roll(self.state,1,0)])
            idx = 0
            mult = 1
            for i in range(self.colors):
                idx += np.sum(neighborhoods==i, axis=0)*mult
                mult *= 4
            #idx = neighborhoods[0,:] + self.colors * neighborhoods[1,:] + self.colors * self.colors * neighborhoods[2,:]
            self.state = self.transitions[idx]
            
    def render(self):
        im = np.zeros((self.history.shape[1], self.history.shape[0], 3)).astype(np.uint8)
        for y in range(self.history.shape[1]):
            for x in range(self.history.shape[0]):
                im[y,x,:] = np.clip(256*self.rgb[self.history[x,y]],0,255).astype(np.uint8)
        im = Image.fromarray(im)
        
        return im


# In[6]:


for carule in range(600,1200):
    ca = TotalCA(colors = 4)
    for example in range(50):
        ca.rgb = np.random.rand(4,3)
        ca.state = np.random.randint(4, size=(128,))
        ca.runca()
        
        im = ca.render()
        im.save("data/%.06d/%.06d.png" % (carule, example))


# In[7]:


for carule in range(1200,1800):
    ca = TotalCA(colors = 5)
    for example in range(50):
        ca.rgb = np.random.rand(5,3)
        ca.state = np.random.randint(5, size=(128,))
        ca.runca()
        
        im = ca.render()
        im.save("data/%.06d/%.06d.png" % (carule, example))


# In[5]:


for carule in range(1800,2400):
    ca = TotalCA(colors = 6)
    for example in range(50):
        ca.rgb = np.random.rand(6,3)
        ca.state = np.random.randint(6, size=(128,))
        ca.runca()
        
        im = ca.render()
        im.save("data/%.06d/%.06d.png" % (carule, example))


# In[ ]:




