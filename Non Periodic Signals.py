
# coding: utf-8

# In[1]:


from __future__ import print_function, division

get_ipython().magic(u'matplotlib inline')

import thinkdsp
import thinkplot
import numpy as np

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets


# In[2]:


signal = thinkdsp.Chirp(start=220, end=880)
wave1 = signal.make_wave(duration=2)
wave1.make_audio()


# In[3]:


wave1.segment(start=0, duration=0.01).plot()


# In[4]:


wave1.segment(start=0.9, duration=0.01).plot()


# In[5]:


def evaluate(self, ts):
    freqs = np.linspace(self.start, self.end, len(ts)-1)
    return self._evaluate(ts, freqs)


# In[6]:


def _evaluate(self, ts, freqs):
    dts = np.diff(ts)
    dphis = PI2 * freqs * dts
    phases = np.cumsum(dphis)
    phases = np.insert(phases, 0, 0)
    ys = self.amp * np.cos(phases)
    return ys


# In[7]:


signal = thinkdsp.ExpoChirp(start=220, end=880)
wave2 = signal.make_wave(duration=2)
wave2.make_audio()


# In[8]:


signal = thinkdsp.ExpoChirp(start=220, end=880)
wave2 = signal.make_wave(duration=2)
wave2.make_audio()


# In[9]:


signal = thinkdsp.SinSignal(freq=440)


# In[10]:


duration = signal.period * 30
wave = signal.make_wave(duration)
wave.plot()


# In[11]:


spectrum = wave.make_spectrum()
spectrum.plot(high=880)
thinkplot.config(xlabel='Frequency (Hz)')


# In[12]:


wave.hamming()
spectrum = wave.make_spectrum()
spectrum.plot(high=880)
thinkplot.config(xlabel = 'Frequency (Hz)')


# In[13]:


signal = thinkdsp.Chirp(start=220, end=440)
wave = signal.make_wave(duration=1)
spectrum = wave.make_spectrum()
spectrum.plot(high=700)
thinkplot.config(xlabel='frequency H(z)')


# In[22]:


def eye_of_sauron(start, end):
    """Plots the spectrum of a chirp.
    
    start: initial frequency
    end: final frequency
    """
    signal = thinkdsp.Chirp(start=start, end=end)
    wave = signal.make_wave(duration=0.5)
    spectrum = wave.make_spectrum()
    
    spectrum.plot(high=1200)
    thinkplot.config(xlabel='frequency (Hz)', ylabel='amplitude')
    


# In[23]:


slider1 = widgets.FloatSlider(min=100, max=1000, value=100, step=50)
slider2 = widgets.FloatSlider(min=100, max=1000, value=200, step=50)
interact(eye_of_sauron, start=slider1, end=slider2);

