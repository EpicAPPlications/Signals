
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


signal = thinkdsp.TriangleSignal(200)
duration = signal.period*3
segment = signal.make_wave(duration, framerate=10000)
segment.plot()
thinkplot.config(ylim=[-1.05, 1.05], legend=False)


# In[3]:


wave = signal.make_wave(duration=0.5, framerate=10000)
wave.apodize()
wave.make_audio()


# In[4]:


spectrum = wave.make_spectrum()
spectrum.plot()


# In[5]:


framerate = 10000
signal = thinkdsp.CosSignal(4500)
duration = signal.period * 5
segment = signal.make_wave(duration, framerate=framerate)
segment.plot()


# In[6]:


framerate = 10000
signal = thinkdsp.CosSignal(5500)
duration = signal.period * 5
segment = signal.make_wave(duration, framerate=framerate)
segment.plot()


# In[7]:


signal = thinkdsp.SawtoothSignal(500)
wave = signal.make_wave(duration=1, framerate=10000)
segment = wave.segment(duration=0.005)
segment.plot()


# In[8]:


hs = np.fft.rfft(wave.ys)
hs


# In[9]:


n = len(wave.ys)
d = 1 / wave.framerate
fs = np.fft.rfftfreq(n, d)
fs


# In[10]:


magnitude = np.absolute(hs)
thinkplot.plot(fs, magnitude)


# In[11]:


angle = np.angle(hs)
thinkplot.plot(fs, angle)


# In[12]:


import random
random.shuffle(angle)
thinkplot.plot(fs, angle)


# In[13]:


i = complex(0, 1)
spectrum = wave.make_spectrum()
spectrum.hs = magnitude * np.exp(i * angle)


# In[14]:


wave2 = spectrum.make_wave()
wave2.normalize()
segment = wave2.segment(duration=0.005)
segment.plot()


# In[15]:


wave2.make_audio()


# In[16]:


wave.make_audio()


# In[17]:


def view_harmonics(freq, framerate):
    signal = thinkdsp.SawtoothSignal(freq)
    wave = signal.make_wave(duration=0.5, framerate=framerate)
    spectrum = wave.make_spectrum()
    spectrum.plot(color='blue')
    thinkplot.show(xlabel='frequency', ylabel='amplitude')
    
    display(wave.make_audio())


# In[18]:


from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

slider1 = widgets.FloatSlider(min=100, max=10000, value=100, step=100)
slider2 = widgets.FloatSlider(min=5000, max=40000, value=10000, step=1000)
interact(view_harmonics, freq=slider1, framerate=slider2)

