
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


signal = thinkdsp.ComplexSinusoid(freq=1, amp=0.6, offset=1)
signal


# In[3]:


wave = signal.make_wave(duration=1, framerate=4)


# In[4]:


wave.ys


# In[5]:


def synthesize1(amps, fs, ts):
    components = [thinkdsp.ComplexSinusoid(freq, amp)
                 for amp, freq in zip(amps, fs)]
    signal = thinkdsp.SumSignal(*components)
    ys = signal.evaluate(ts)
    return ys


# In[6]:


amps = np.array([0.6, 0.25, 0.1, 0.05])
fs = [100, 200, 300, 400]
framerate = 11025
ts = np.linspace(0, 1, framerate)
ys = synthesize1(amps, fs, ts)
ys


# In[7]:


n = 500
thinkplot.plot(ts[:n], ys[:n].real, label='real')
thinkplot.plot(ts[:n], ys[:n].imag, label='imag')


# In[8]:


PI2 = np.pi * 2


# In[9]:


def synthesize2(amps, fs, ts):
    args = np.outer(ts, fs)
    M = np.exp(1j * PI2 * args)
    ys = np.dot(M, amps)
    return ys


# In[10]:


ys = synthesize2(amps, fs, ts)
ys


# In[11]:


phi = 1.5
amps2 = amps * np.exp(1j * phi)
ys2 = synthesize2(amps, fs, ts)

thinkplot.plot(ts[:n], ys.real[:n])
thinkplot.plot(ts[:n], ys.imag[:n])


# In[12]:


def analyze1(ys, fs, ts):
    args = np.outer(ts, fs)
    M = np.exp(1j * PI2 * args)
    amps = np.linalg.solve(M, ys)
    return amps


# In[13]:


n = len(fs)


# In[14]:


amps2 = analyze1(ys[:n], fs, ts[:n])


# In[15]:


amps2


# In[16]:


N = 4
ts = np.arange(N) / N
fs = np.arange(N)
args = np.outer(ts, fs)
M = np.exp(1j * PI2 * args)


# In[17]:


MstarM = M.conj().transpose().dot(M)


# In[18]:


def analyze2(ys, ts, fs):
    args = np.outer(ts, fs)
    M = np.exp(1j * PI2 * args)
    amps = M.conj().transpose().dot(ys) / N
    return amps


# In[21]:


N = 4
amps = np.array([0.6, 0.25, 0.1, 0.05])
fs = np.arange(N)
ts = np.arange(N) / N
ys = synthesize2(amps, fs, ts)
amps3 = analyze2(ys, fs, ts)
amps3


# In[22]:


def synthesis_matrix(N):
    ts = np.arange(N) / N
    fs = np.arange(N)
    args = np.outer(ts, fs)
    M = np.exp(1j * PI2 * args)
    return M


# In[23]:


def analyze3(ys):
    N = len(ys)
    M = synthesis_matrix(N)
    amps = M.conj().transpose().dot(ys) / M
    return amps


# In[26]:


def dft(ys):
    N = len(ys)
    M = synthesis_matrix(N)
    amps = M.conj().transpose().dot(ys)
    return amps


# In[27]:


dft(ys)


# In[28]:


np.fft.fft(ys)


# In[29]:


def idft(ys):
    N = len(ys)
    M = synthesis_matrix(N)
    amps = M.dot(ys) / N
    return amps


# In[30]:


ys = idft(amps)


# In[31]:


dft(ys)


# In[32]:


signal = thinkdsp.SawtoothSignal(freq=500)
wave = signal.make_wave(duration=0.1, framerate=10000)
hs = dft(wave.ys)
amps = np.absolute(hs)


# In[33]:


fs = np.arange(N)


# In[34]:


fs = np.arange(N) * framerate / N

