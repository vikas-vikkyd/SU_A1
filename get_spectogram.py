import pandas as pd
import librosa
import scipy.signal as signal
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

audio_file_path = "UrbanSound8K/audio"
window_size = 1024
hop_size = window_size // 2


def read_sound_files(audio_file_path):
    """Function to read sound files and their metadata"""
    # read sound files
    data = []
    for folder in os.listdir(audio_file_path):
        if folder != ".DS_Store":
            for audio_file in os.listdir(os.path.join(audio_file_path, folder)):
                file = os.path.join(audio_file_path, folder, audio_file)
                try:
                    audio_signal, fs = librosa.load(file, sr=None)
                except Exception as e:
                    print("Not able to read file {}".format(audio_file))
                data.append([audio_signal, fs, audio_file])
    return data


def save_spectrogram(f, t, Zxx, file_path):
    """Save the spectrogram."""
    plt.pcolormesh(t, f, np.abs(Zxx), shading="auto")
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.title("STFT")
    plt.savefig(file_path)
    return None


def compute_stft(
    audio_signal, fs, window, window_size, hop_size, file_path, save=False
):
    # Compute the STFT using scipy's spectrogram function
    f, t, Zxx = signal.stft(
        audio_signal, fs, window=window, nperseg=window_size, noverlap=hop_size
    )

    # save spectrogram
    if save:
        save_spectrogram(f, t, np.log(Zxx), file_path)

    # save intesity magnitue
    Zxx_abs = np.abs(Zxx)
    max_aplitude = list(Zxx_abs.max(axis=1))
    max_aplitude.append(file_path.split("/")[-1])
    return max_aplitude


def main():
    """main module to prepare data for spectrogram and visualize them"""
    data = read_sound_files(audio_file_path)

    # create directory for spectrogram
    spectrogram_dir = os.path.join(os.curdir, "spectrogram")
    if not os.path.exists(spectrogram_dir):
        os.makedirs(spectrogram_dir)
        os.makedirs(os.path.join(spectrogram_dir, "hann"))
        os.makedirs(os.path.join(spectrogram_dir, "hamming"))
        os.makedirs(os.path.join(spectrogram_dir, "rectangular"))

    # Define all the window
    hanning_window = np.hanning(window_size)
    hamming_window = np.hamming(window_size)
    rectangular_window = signal.windows.boxcar(window_size)

    # save spectrogram for some random audio
    random_number = random.sample(range(0, len(data) - 1), 10)
    for i in tqdm(random_number):
        audio_signal, fs, audio_file = data[i]
        # Apply the Hanning window
        hanning_window = np.hanning(window_size)
        hamming_window = np.hamming(window_size)
        rectangular_window = signal.windows.boxcar(window_size)

        # save spectrograme for hanning window
        file_path = os.path.join(
            spectrogram_dir, "hann", audio_file.replace(".wav", ".png")
        )
        compute_stft(
            audio_signal,
            fs,
            hanning_window,
            window_size,
            hop_size,
            file_path,
            save=True,
        )

        # save spectrograme for hamming window
        file_path = os.path.join(
            spectrogram_dir, "hamming", audio_file.replace(".wav", ".png")
        )
        compute_stft(
            audio_signal,
            fs,
            hamming_window,
            window_size,
            hop_size,
            file_path,
            save=True,
        )

        # save spectrograme for rectangular window
        file_path = os.path.join(
            spectrogram_dir, "rectangular", audio_file.replace(".wav", ".png")
        )
        compute_stft(
            audio_signal,
            fs,
            rectangular_window,
            window_size,
            hop_size,
            file_path,
            save=True,
        )
    
    # generate dataset with spectrogram features which will be used for classification
    train_data = []
    for audio_signal, fs, audio_file in tqdm(data):
        file_path = os.path.join(
            spectrogram_dir, "train_data", audio_file.replace(".wav", ".png")
        )
        max_aplitude = compute_stft(
            audio_signal,
            fs,
            hanning_window,
            window_size,
            hop_size,
            file_path
        )
        train_data.append(max_aplitude)

    # save data to csv file
    train_data = np.array(train_data)
    columns = ["f" + str(i) for i in range(train_data.shape[1]-1)]
    columns.append("file_path")
    df_train_data = pd.DataFrame(data=train_data, columns=columns)
    df_train_data.to_csv("train_data.csv", index=False)


if __name__ == "__main__":
    main()
