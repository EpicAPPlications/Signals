
# coding: utf-8

# In[1]:


from __future__ import print_function, division

import thinkdsp
import thinkplot

import numpy as np

import warnings
warnings.filterwarnings('ignore')

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

get_ipython().magic(u'matplotlib inline')
PI2 = np.pi * 2


# In[2]:


def synthesize1(amps, fs, ts):
    components = [thinkdsp.CosSignal(freq, amp)
                  for amp, freq in zip(amps, fs)]
    signal = thinkdsp.SumSignal(*components)

    ys = signal.evaluate(ts)
    return ys


# In[3]:


amps = np.array([0.6, 0.25, 0.1, 0.05])
fs = [100, 200, 300, 400]
framerate = 11025

ts = np.linspace(0, 1, framerate)
ys = synthesize1(amps, fs, ts)
wave = thinkdsp.Wave(ys, ts, framerate)
wave.apodize()
wave.make_audio()


# In[4]:


def synthesize2(amps, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    ys = np.dot(M, amps)
    return ys


# In[5]:


ys = synthesize2(amps, fs, ts)
wave = thinkdsp.Wave(ys, framerate)
wave.apodize()
wave.make_audio()


# In[6]:


ys1 = synthesize1(amps, fs, ts)
ys2 = synthesize2(amps, fs, ts)
max(abs(ys1 - ys2))


# In[7]:


def analyze1(ys, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    amps = np.linalg.solve(M, ys)
    return amps


# In[8]:


n = len(fs)
amps2 = analyze1(ys[:n], fs, ts[:n])
amps2


# In[9]:


np.set_printoptions(precision=3, suppress=True)


# In[11]:


def test1():
    amps = np.array([0.6, 0.25, 0.1, 0.05])
    N = 4.0
    time_unit = 0.001
    ts = np.arange(N) / N* time_unit
    max_freq = N / time_unit / 2
    fs = np.arange(N) / N * max_freq
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    return M

M = test1()
M


# In[12]:


M.transpose().dot(M)


# In[13]:


def test2():
    amps = np.array([0.6, 0.25, 0.1, 0.05])
    N = 4.0
    ts = (0.5 + np.arange(N)) / N
    fs = (0.5 + np.arange(N)) / 2
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    return M
    
M = test2()
M


# In[14]:


M.transpose().dot(M)


# In[15]:


def analyze2(ys, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    amps = M.dot(ys) / 2
    return amps


# In[16]:


n = len(fs)
amps2 = analyze1(ys[:n], fs, ts[:n])
amps2


# In[17]:


def dct_iv(ys):
    N = len(ys)
    ts = (0.5 + np.arange(N)) / N
    fs = (0.5 + np.arange(n)) / 2
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    amps = np.dot(M, ys) / 2
    return amps


# In[18]:


amps = np.array([0.6, 0.25, 0.1, 0.05])
N = 4.0
ts = (0.5 + np.arange(N)) / N
fs = (0.5 + np.arange(N)) / 2
ys = synthesize2(amps, fs, ts)

amps2 = dct_iv(ys)
print(max(abs(amps - amps2)))



# In[19]:


def inverse_dct_iv(amps):
    return dct_iv(amps) * 2


# In[20]:


amps = [0.6, 0.25, 0.1, 0.05]
ys = inverse_dct_iv(amps)
amps2 = dct_iv(ys)
print(max(abs(amps - amps2)))


# In[21]:


signal = thinkdsp.TriangleSignal(freq=400)
wave = signal.make_wave(duration=1.0, framerate=10000)
wave.make_audio()


# In[22]:


dct = wave.make_dct()
dct.plot()
thinkplot.config(xlabel='Frequency (Hz)', ylabel='DCT')


# In[23]:


wave2 = dct.make_wave()


# In[25]:


max(abs(wave.ys-wave2.ys))


# In[26]:


signal = thinkdsp.TriangleSignal(freq=400, offset=0)
wave = signal.make_wave(duration=1.0, framerate=10000)
wave.ys *= -1
wave.make_dct().plot()


# In[30]:


signal = thinkdsp.TriangleSignal(freq=400, offset=np.pi)
wave = signal.make_wave(duration=1.0, framerate=10000)
wave.make_dct().plot()

