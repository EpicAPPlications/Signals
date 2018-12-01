import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

Nf = 64
fs = 64
f = 10
t = np.arange(0,1,1/2.)
deltaf = 1/2.
pi = 3.14

#keep x and y-axes on same respective scale
fig,ax = plt.subplots(2,1,sharex=True,sharey=True)
fig.set_size_inches((8,3))

x = np.cos(2*pi*f*t) + np.cos(2*pi*(f*2*t))
X = fft.fft(x, Nf)/np.sqrt(Nf)
ax[0].plot(np.linspace(0, fs, Nf),abs(X), '-o')
ax[0].set_title(r'$\delta f = 2$ Hz, $T=1$ s', fontsize=18)
ax[0].set_ylabel(r'$|X(f)|$', fontsize=18)
ax[0].grid()

x = np.cos(2*pi*f*t) + np.cos(2*pi*(f+deltaf)*t)
X = fft.fft(x, Nf)/np.sqrt(Nf)
ax[1].plot(np.linspace(0, fs, Nf),abs(X),'-o')
ax[1].set_title(r'$\delta f = 1/2$ Hz, $T=$1 s',fontsize=14)
ax[1].set_ylabel(r'$|X(f)|$', fontsize=18)

ax[1].set_xlabel('Frequency (Hz)', fontsize=18)
ax[1].set_title(r'$\delta f = 1/2$ Hz, $T=1$ s',fontsize=14)
ax[1].set_ylabel(r'$|X(f)|$',fontsize=18)
