
# coding: utf-8

# In[8]:


from __future__ import print_function, division

get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings('ignore')

import thinkdsp
import thinkplot
import thinkstats2 
import scipy
import numpy as np

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets


# In[9]:


signal = thinkdsp.UncorrelatedUniformNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
wave.make_audio()


# In[10]:


segment = wave.segment(duration=0.1)
segment.plot(linewidth=1)
thinkplot.config(xlabel='time',
                ylabel='amplitude',
                ylim=[-1.05, 1.05],
                legend=False)


# In[17]:


spectrum = wave.make_spectrum()
spectrum.plot(linewidth=0.5)
thinkplot.config(xlabel='frequency (Hz)',
                 ylabel='amplitude',
                 xlim=[0, spectrum.fs[-1]])


# In[11]:


signal =  thinkdsp.BrownianNoise()
wave =  signal.make_wave(duration=0.5,  framerate=11025)
wave.plot()


# In[12]:


spectrum = wave.make_spectrum()
spectrum.plot_power(linewidth=1, alpha=0.5)
thinkplot.config(xscale='log', yscale='log')


# In[13]:


def estimate_slope(self):
    x = np.log(self.fs[1:])
    y = np.log(self.power[1:])
    t = scipy.stats.linregress(x, y)
    return t


# In[14]:


estimate_slope(spectrum)


# In[15]:


spectrum = wave.make_spectrum()
spectrum.plot(linewidth=0.5)
thinkplot.config(xlabel='frequency (Hz)',
                 ylabel='amplitude',
                 xlim=[0, spectrum.fs[-1]])



# In[18]:


integ = spectrum.make_integrated_spectrum()
integ.plot_power()
thinkplot.config(xlabel='frequency (Hz)',
                ylabel='cumulative power',
                xlim=[0, spectrum.fs[-1]])


# In[19]:


signal = thinkdsp.BrownianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
wave.make_audio()


# In[21]:


wave.plot(linewidth=1)
thinkplot.config(xlabel='time',
                 ylabel='amplitude',
                 ylim=[-1.05, 1.05])


# In[22]:


spectrum.hs[0] = 0

spectrum.plot_power(linewidth=0.5)
thinkplot.config(xlabel='frequency (Hz)',
                 ylabel='power',
                 xscale='log',
                 yscale='log',
                 xlim=[0, spectrum.fs[-1]])


# In[23]:


signal = thinkdsp.BrownianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
spectrum = wave.make_spectrum()
result = spectrum.estimate_slope()
result.slope


# In[24]:


signal = thinkdsp.PinkNoise(beta=1)
wave = signal.make_wave(duration=0.5)
wave.make_audio()


# In[25]:


colors = ['#9ecae1', '#4292c6', '#2171b5']
betas = [0, 1, 2]

for beta, color in zip(betas, colors):
    signal = thinkdsp.PinkNoise(beta=beta)
    wave = signal.make_wave(duration=0.5, framerate=1024)
    spectrum = wave.make_spectrum()
    spectrum.hs[0] = 0
    spectrum.plot_power(linewidth=1, color=color)
    
thinkplot.config(xlabel='frequency (Hz)',
                 ylabel='power',
                 xscale='log',
                 yscale='log',
                 xlim=[0, spectrum.fs[-1]])


# In[26]:


signal = thinkdsp.UncorrelatedGaussianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
wave.plot(linewidth=0.5)
thinkplot.config(xlabel='time',
                 ylabel='amplitude')


# In[27]:


spectrum = wave.make_spectrum()
spectrum.plot_power(linewidth=1)
thinkplot.config(xlabel='frequency (Hz)',
                 ylabel='power',
                 xlim=[0, spectrum.fs[-1]])


# In[28]:


from thinkstats2 import NormalProbabilityPlot

NormalProbabilityPlot(spectrum.real, label='real part')
thinkplot.config(xlabel='normal sample',
                 ylabel='power',
                 ylim=[-250, 250],
                 legend=True,
                 loc='lower right')


# In[29]:


NormalProbabilityPlot(spectrum.imag, label='imag part')
thinkplot.config(xlabel='normal sample',
                 ylabel='power',
                 ylim=[-250, 250],
                 legend=True,
                 loc='lower right')

