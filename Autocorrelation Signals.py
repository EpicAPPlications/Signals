
# coding: utf-8

# In[1]:


from __future__ import print_function, division

import thinkdsp
import thinkplot
import thinkstats2 

import numpy as np

import warnings
warnings.filterwarnings('ignore')

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

PI2 = np.pi * 2

get_ipython().magic(u'matplotlib inline')


# In[2]:


def make_sine(offset):
    signal = thinkdsp.SinSignal(freq=440, offset=offset)
    wave = signal.make_wave(duration=0.5, framerate=10000)
    return wave


# In[3]:


wave1 = make_sine(offset=0)
wave2 = make_sine(offset=1)

thinkplot.preplot(2)
wave1.segment(duration=0.01).plot()
wave2.segment(duration=0.01).plot()
thinkplot.config(xlabel='Time (s)', ylim=[-1.05, 1.05])


# In[4]:


print(np.corrcoef(wave1.ys, wave2.ys))


# In[5]:


wave1.corr(wave2)


# In[8]:


def compute_corr(offset):
    wave1 = make_sine(offset=0)
    wave2 = make_sine(offset=-offset)
    
    thinkplot.preplot(2)
    wave1.segment(duration=0.01).plot()
    wave2.segment(duration=0.01).plot()
    
    corr = wave1.corr(wave2)
    print('corr =', corr)
    
    thinkplot.config(xlabel='Time (s)', ylim=[-1.05, 1.05])


# In[9]:


slider = widgets.FloatSlider(min=0, max=PI2, value=1)
interact(compute_corr, offset=slider);


# In[10]:


offsets = np.linspace(0, PI2, 101)

corrs = []
for offset in offsets:
    wave2 = make_sine(offset)
    corr = np.corrcoef(wave1.ys, wave2.ys)[0, 1]
    corrs.append(corr)
    
thinkplot.plot(offsets, corrs)
thinkplot.config(xlabel='Offset (radians)', 
                ylabel='Correlation',
                axis=[0, PI2, -1.05, 1.05])


# In[11]:


def serial_corr(wave, lag=1):
    N = len(wave)
    y1 = wave.ys[lag:]
    y2 = wave.ys[:N-lag]
    corr = np.corrcoef(y1, y2, ddof=0)[0, 1]
    return corr


# In[12]:


signal = thinkdsp.UncorrelatedGaussianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
serial_corr(wave)


# In[13]:


signal = thinkdsp.BrownianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
serial_corr(wave)


# In[14]:


signal = thinkdsp.PinkNoise(beta=1)
wave = signal.make_wave(duration=0.5, framerate=11025)
serial_corr(wave)


# In[17]:


def autocorr(wave):
    """Computes and plots the autocorrelation function.

    wave: Wave
    """
    lags = range(len(wave.ys)//2)
    corrs = [serial_corr(wave, lag) for lag in lags]
    return lags, corrs


# In[18]:


def plot_pink_autocorr(beta, label):
    signal = thinkdsp.PinkNoise(beta=beta)
    wave = signal.make_wave(duration=1.0, framerate=10000)
    lags, corrs = autocorr(wave)
    thinkplot.plot(lags, corrs, label=label)



# In[19]:


np.random.seed(19)
thinkplot.preplot(3)

for beta in [1.7, 1.0, 0.3]:
    label = r'$\beta$ = %.1f' % beta
    plot_pink_autocorr(beta, label)

thinkplot.config(xlabel='Lag',
                 ylabel='Correlation',
                 xlim=[-1, 1000],
                 ylim=[-0.05, 1.05],
                 legend=True)


# In[21]:


spectrum = wave.make_spectrum()
spectrum.plot()
thinkplot.config(xlabel='Frequency (Hz)', ylabel='Amplitude')


# In[23]:


duration = 0.01
segment = wave.segment(start=0.2, duration=duration)
segment.plot()
thinkplot.config(xlabel='Time (s)', ylim=[-1, 1])


# In[24]:


spectrum = segment.make_spectrum()
spectrum.plot(high=1000)
thinkplot.config(xlabel='Frequency (Hz)', ylabel='Amplitude')


# In[25]:


len(segment), segment.framerate, spectrum.freq_res


# In[26]:


def plot_shifted(wave, offset=0.001, start=0.2):
    thinkplot.preplot(2)
    segment1 = wave.segment(start=start, duration=0.01)
    segment1.plot(linewidth=2, alpha=0.8)

    # start earlier and then shift times to line up
    segment2 = wave.segment(start=start-offset, duration=0.01)
    segment2.shift(offset)
    segment2.plot(linewidth=2, alpha=0.4)

    corr = segment1.corr(segment2)
    text = r'$\rho =$ %.2g' % corr
    thinkplot.text(segment1.start+0.0005, -0.8, text)
    thinkplot.config(xlabel='Time (s)', xlim=[start, start+duration], ylim=[-1, 1])

plot_shifted(wave, 0.0001)


# In[27]:


end = 0.004
slider1 = widgets.FloatSlider(min=0, max=end, step=end/40, value=0)
slider2 = widgets.FloatSlider(min=0.1, max=0.5, step=0.05, value=0.2)
interact(plot_shifted, wave=fixed(wave), offset=slider1, start=slider2)
None


# In[28]:


lags, corrs = autocorr(segment)
thinkplot.plot(lags, corrs)
thinkplot.config(xlabel='Lag (index)', ylabel='Correlation', ylim=[-1, 1])


# In[32]:


N = len(segment)
corrs2 = np.correlate(segment.ys, segment.ys, mode='same')
lags = np.arange(-N//2, N//2)
thinkplot.plot(lags, corrs2)
thinkplot.config(xlabel='Lag', ylabel='Correlation', xlim=[-N//2, N//2])


# In[33]:


N = len(corrs2)
lengths = range(N, N//2, -1)

half = corrs2[N//2:].copy()
half /= lengths
half /= half[0]
thinkplot.plot(half)
thinkplot.config(xlabel='Lag', ylabel='Correlation', ylim=[-1.05, 1.05])


# In[34]:


thinkplot.preplot(2)
thinkplot.plot(half)
thinkplot.plot(corrs)
thinkplot.config(xlabel='Lag', ylabel='Correlation', ylim=[-1.05, 1.05])

