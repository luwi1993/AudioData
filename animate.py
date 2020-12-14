import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import sys
def animate_stft(stft, freqs, duration):
    n_freqs, n_steps = stft.shape
    max_ =  np.max(np.abs(stft))
    fig = plt.figure()
    def buildmebarchart(i=int):
        plt.clf()
        plt.xlim(0, n_freqs)
        plt.ylim(0, max_)
        plt.xlabel('freq')
        plt.title("stft in interval " + str(np.round(duration/n_steps * i, 2))  + "sec. to " + str(np.round(duration/n_steps * (i+1), 2)) + " sec.")
        p = plt.plot(freqs, np.abs(stft[:, i]))
    animator = ani.FuncAnimation(fig, buildmebarchart, interval=duration/n_steps*1000, save_count=n_steps, repeat=True)
    animator.save('files/stft.gif')

