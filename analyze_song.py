import os
import librosa
import numpy as np
from get_spectogram import save_spectrogram
import scipy.signal as signal

song1 = "songs/twisterion-b1-221376.mp3"
song2 = "songs/eterna-cancao-wav-12569.mp3"
song3 = "songs/sci-fi-background-258999.mp3"
song4 = "songs/see-you-later-203103.mp3"
window_size = 1024
hop_size = window_size // 2


def analyze_song(file):
    """Module to analyze song file"""
    audio_signal, fs = librosa.load(file, sr=None)

    # define window
    window = np.hanning(window_size)

    # Compute the STFT using scipy's spectrogram function
    f, t, Zxx = signal.stft(
        audio_signal, fs, window=window, nperseg=window_size, noverlap=hop_size
    )
    file_path = os.path.join(
        os.curdir, "spectrogram", file.split("/")[-1].replace(".mp3", ".png")
    )
    save_spectrogram(f, t, np.log(Zxx), file_path)


# save spectrograme for song1
analyze_song(song1)
analyze_song(song2)
analyze_song(song3)
analyze_song(song4)
