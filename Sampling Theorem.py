
# coding: utf-8

# In[1]:


from __future__ import print_function, division

get_ipython().magic(u'matplotlib inline')

import thinkdsp
import thinkplot

import numpy as np

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import display


# In[2]:


cos_sig = thinkdsp.CosSignal(freq=440, amp=1.0, offset=0)
sin_sig = thinkdsp.SinSignal(freq=880, amp=0.5, offset=0)


# In[3]:


cos_sig.plot()
thinkplot.config(xlabel='Time (s)')


# In[4]:


sin_sig.plot()
thinkplot.config(xlabel='Time (s)')


# In[5]:


mix = sin_sig + cos_sig
mix


# In[6]:


mix.plot()


# In[7]:


wave = mix.make_wave(duration=0.5, start=0, framerate=11025)
wave


# In[8]:


from IPython.display import Audio
audio = Audio(data=wave.ys, rate=wave.framerate)
audio


# In[9]:


wave.make_audio()


# In[10]:


print('Number of samples', len(wave.ys))
print('Timestep in ms', 1 / wave.framerate * 1000)


# In[11]:


period = mix.period
segment = wave.segment(start=0, duration=period*3)
period


# In[12]:


segment.plot()
thinkplot.config(xlabel='Time (s)')


# In[13]:


wave.normalize()
wave.apodize()
wave.plot()
thinkplot.config(xlabel='Time (s)')


# In[14]:


wave.write('temp.wav')


# In[15]:


thinkdsp.play_wave(filename='temp.wav', player='aplay')


# In[16]:


wave = thinkdsp.read_wave('temp.wav')


# In[17]:


wave.make_audio()


# In[22]:


start = 0
duration = 0.6
segment = wave.segment(start, duration)
segment.plot()
thinkplot.config(xlabel='Time (s)')


# In[23]:


wave.plot()


# In[24]:


segment.plot()

