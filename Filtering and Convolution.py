
# coding: utf-8

# In[1]:


from __future__ import print_function, division

get_ipython().magic(u'matplotlib inline')

import thinkdsp
import thinkplot
import thinkstats2

import numpy as np
import pandas as pd
import scipy.signal

np.set_printoptions(precision=3, suppress=True)


# In[2]:


PI2 = 2 * np.pi
GRAY = '0.7'


# In[3]:


signal = thinkdsp.SquareSignal(freq=440)
wave = signal.make_wave(duration=1, framerate=4400)
segment = wave.segment(duration=0.01)


# In[4]:


window = np.ones(11)
window /= sum(window)


# In[5]:


ys = segment.ys
N = len(ys)
padded = thinkdsp.zero_pad(window, N)


# In[6]:


prod = padded * ys


# In[7]:


sum(ys)


# In[8]:


convolved = np.convolve(ys, window, mode='valid')
smooth2 = thinkdsp.Wave(convolved, framerate=wave.framerate)
smooth2


# In[9]:


spectrum = wave.make_spectrum()
spectrum.plot(color='GRAY')


# In[10]:


convolved = np.convolve(wave.ys, window, mode='same')
smooth = thinkdsp.Wave(convolved, framerate=wave.framerate)
spectrum2 = smooth.make_spectrum()
spectrum2.plot()


# In[11]:


amps = spectrum.amps
amps2 = spectrum2.amps
ratio = amps2 / amps
ratio[amps<560] = 0
thinkplot.plot(ratio)


# In[12]:


padded = zero_pad(window, N)
dft_window = np.fft.rfft(padded)
thinkplot.plot(abs(dft_window))


# In[ ]:


gaussian =  scipy.signal.gaussian(M=11, std=2)
gaussian /= sum(gaussian)


# In[ ]:


names = ['date', 'open', 'high', 'low', 'close', 'volume']
df = pd.read_csv('C:\Users\esehu\Downloads\FB.csv', header=0, names=names, parse_dates=[0])
df.head()


# In[ ]:


signal = thinkdsp.SawtoothSignal(freq=440)
wave = signal.make_wave(duration=1.0, framerate=44100)
wave.make_audio()


# In[ ]:


window = np.ones(11)
window /= sum(window)
thinkplot.plot(window)


# In[ ]:


segment = wave.segment(duration=0.01)
segment.plot()
thinkplot.config(xlabel='Time (s)', ylim=[-1.05, 1.05])


# In[ ]:


N = len(segment)
padded = thinkdsp.zero_pad(window, N)
thinkplot.plot(padded)
thinkplot.config(xlabel='Index')


# In[ ]:


prod = padded * segment.ys
print(sum(prod))


# In[ ]:


smoothed = np.zeros(N)
rolled = padded.copy()
for i in range(N):
    smoothed[i] = sum(rolled * segment.ys)
    rolled = np.roll(rolled, 1)


# In[ ]:


segment.plot(color=GRAY)
smooth = thinkdsp.Wave(smoothed, framerate=wave.framerate)
smooth.plot()
thinkplot.config(xlabel='Time(s)', ylim=[-1.05, 1.05])


# In[ ]:


segment.plot(color=GRAY)
ys = np.convolve(segment.ys, window, mode='valid')
smooth2 = thinkdsp.Wave(ys, framerate=wave.framerate)
smooth2.plot()
thinkplot.config(xlabel='Time(s)', ylim=[-1.05, 1.05])


# In[ ]:


convolved = np.convolve(wave.ys, window, mode='same')
smooth = thinkdsp.Wave(convolved, framerate=wave.framerate)
smooth.make_audio()


# In[ ]:


spectrum = wave.make_spectrum()
spectrum.plot(color=GRAY)

spectrum2 = smooth.make_spectrum()
spectrum2.plot()

thinkplot.config(xlabel='Frequency (Hz)',
                 ylabel='Amplitude',
                 xlim=[0, 22050])


# In[ ]:


amps = spectrum.amps
amps2 = spectrum2.amps
ratio = amps2 / amps    
ratio[amps<280] = 0

thinkplot.plot(ratio)
thinkplot.config(xlabel='Frequency (Hz)',
                     ylabel='Amplitude ratio',
                     xlim=[0, 22050])


# In[ ]:


padded = thinkdsp.zero_pad(window, len(wave))
dft_window = np.fft.rfft(padded)

thinkplot.plot(abs(dft_window), color=GRAY, label='DFT(window)')
thinkplot.plot(ratio, label='amplitude ratio')

thinkplot.config(xlabel='Frequency (Hz)',
                     ylabel='Amplitude ratio',
                     xlim=[0, 22050], loc='upper right')


# In[ ]:


boxcar = np.ones(11)
boxcar /= sum(boxcar)


# In[ ]:


gaussian = scipy.signal.gaussian(M=11, std=2)
gaussian /= sum(gaussian)


# In[ ]:


thinkplot.preplot(2)
thinkplot.plot(boxcar, label='boxcar')
thinkplot.plot(gaussian, label='Gaussian')
thinkplot.config(xlabel='Index',
                 loc='upper right')


# In[ ]:


ys = np.convolve(wave.ys, gaussian, mode='same')
smooth = thinkdsp.Wave(ys, framerate=wave.framerate)
spectrum2 = smooth.make_spectrum()


# In[ ]:


amps = spectrum.amps
amps2 = spectrum2.amps
ratio = amps2 / amps    
ratio[amps<560] = 0


# In[ ]:


padded = thinkdsp.zero_pad(gaussian, len(wave))
dft_gaussian = np.fft.rfft(padded)


# In[ ]:


thinkplot.plot(abs(dft_gaussian), color='0.7', label='Gaussian filter')
thinkplot.plot(ratio, label='amplitude ratio')

thinkplot.config(xlabel='Frequency (Hz)',
                 ylabel='Amplitude ratio',
                 xlim=[0, 22050])


# In[ ]:


def plot_filter(M=11, std=2):
    signal = thinkdsp.SquareSignal(freq=440)
    wave = signal.make_wave(duration=1, framerate=44100)
    spectrum = wave.make_spectrum()

    gaussian = scipy.signal.gaussian(M=M, std=std)
    gaussian /= sum(gaussian)
    high = gaussian.max()
    
    thinkplot.preplot(cols=2)
    thinkplot.plot(gaussian)
    thinkplot.config(xlabel='Index', ylabel='Window', 
                     xlim=[0, len(gaussian)-1], ylim=[0, 1.1*high])

    ys = np.convolve(wave.ys, gaussian, mode='same')
    smooth = thinkdsp.Wave(ys, framerate=wave.framerate)
    spectrum2 = smooth.make_spectrum()

    # plot the ratio of the original and smoothed spectrum
    amps = spectrum.amps
    amps2 = spectrum2.amps
    ratio = amps2 / amps    
    ratio[amps<560] = 0

    # plot the same ratio along with the FFT of the window
    padded = thinkdsp.zero_pad(gaussian, len(wave))
    dft_gaussian = np.fft.rfft(padded)

    thinkplot.subplot(2)
    thinkplot.plot(abs(dft_gaussian), color=GRAY, label='Gaussian filter')
    thinkplot.plot(ratio, label='amplitude ratio')

    thinkplot.show(xlabel='Frequency (Hz)',
                     ylabel='Amplitude ratio',
                     xlim=[0, 22050],
                     ylim=[0, 1.05])


# In[15]:


plot_filter


# In[13]:


plot_filter(M=11, std=2)


# In[14]:


from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

slider = widgets.IntSlider(min=2, max=100, value=11)
slider2 = widgets.FloatSlider(min=0, max=20, value=2)
interact(plot_filter, M=slider, std=slider2);

