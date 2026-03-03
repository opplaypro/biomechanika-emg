from typing import Tuple
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import re


def load_data(
        file_path: str | Path
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data from a specified file path, clears it, and formats it for use
    in plotting and analysis.

    Parameters
    ----------
    file_path : str | Path
        The path to the file containing the data to be cleared and formatted.

    Returns
    -------
    pd.DataFrame:
        A DataFrame containing the session statistics extracted from the file.
    pd.DataFrame:
        A DataFrame containing the markers extracted from the file.
    pd.DataFrame:
        A DataFrame containing the EMG data extracted from the file.

    """
    file_headers = ["SESSION STATISTICS", "MARKERS", "EMG"]
    statistics = ""
    markers = ""
    data = ""
    current_header = ""

    with open(file_path, "r", encoding="utf-16") as file:
        lines = file.readlines()
    for line in lines:
        if line.strip() in file_headers:
            current_header = line.strip()
            continue
        if current_header == "SESSION STATISTICS":
            statistics += line.rstrip() + "\n"
        elif current_header == "MARKERS":
            markers += line.rstrip() + "\n"
        elif current_header == "EMG":
            data += line.rstrip() + "\n"

    # Convert strings to DataFrames
    statistics_df = pd.DataFrame([row.split("\t")
                                  for row in statistics.strip().split("\n")])
    markers_df = pd.DataFrame([row.split("\t")
                               for row in markers.strip().split("\n")])
    data_df = pd.DataFrame([row.split("\t")
                            for row in data.strip().split("\n")])

    return statistics_df, markers_df, data_df


def preprocess_data(
        data: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Preprocesses the EMG data for analysis and plotting.

    Parameters
    ----------
    data : pd.DataFrame
        The EMG data to be preprocessed.

    Returns
    -------
    pd.DataFrame
        The preprocessed EMG data.
    """
    # move the first row to the header
    data.columns = pd.Index(data.iloc[0])

    # convert values to numeric only in columns 1 and 2, ignoring errors
    cols = data.columns[1:3]
    data[cols] = data[cols].replace(',', '.', regex=True).apply(
        pd.to_numeric, errors='coerce')

    return data


def plot_data(
        data_1: pd.Series,
        data_2: pd.Series,
        strings: list[str],
        fs: float,
        window_size: int = 25
        ) -> None:
    """
    Plots the EMG data.

    Parameters
    ----------
    data_1 : pd.Series
        The first channel to be plotted.
    data_2 : pd.Series
        The second channel to be plotted.
    strings : list[str]
        A list of strings to be used in the plot title and labels.
    """
    moving_avg_1 = data_1.rolling(window=window_size).mean()
    moving_avg_2 = data_2.rolling(window=window_size).mean()
    rms_1 = np.sqrt((data_1**2).rolling(window=window_size).mean())
    rms_2 = np.sqrt((data_2**2).rolling(window=window_size).mean())

    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    time = np.arange(len(data_1)) / fs

    axs[0].plot(time, data_1,
                label="channel 1 (raw)")
    axs[0].plot(time, data_2,
                label="channel 2 (raw)")
    axs[1].plot(time[:len(moving_avg_1)], moving_avg_1,
                label="channel 1 (moving avg)")
    axs[1].plot(time[:len(moving_avg_2)], moving_avg_2,
                label="channel 2 (moving avg)")
    axs[2].plot(time[:len(rms_1)], rms_1,
                label="channel 1 (RMS)")
    axs[2].plot(time[:len(rms_2)], rms_2,
                label="channel 2 (RMS)")
    axs[0].set_title(f"{strings[0]} - Raw Data")
    axs[1].set_title(f"{strings[0]} - Moving Average ({window_size})")
    axs[2].set_title(f"{strings[0]} - RMS ({window_size})")

    for ax in axs:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EMG Signal (µv)")
        ax.legend()
    plt.tight_layout()
    plt.show()


def stft_plot(
        data: pd.Series,
        fs: float,
        strings: list[str]
        ) -> None:
    """
    Plots the Short-Time Fourier Transform (STFT) of the EMG data.
    Parameters
    ----------
    data : pd.Series
        The EMG data to be analyzed and plotted.
    fs : int
        The sampling frequency of the EMG data.
    strings : list[str]
        A list of strings to be used in the plot title and labels.
    """
    f, t, Zxx = signal.stft(data, fs=fs, nperseg=256)
    plt.figure(figsize=(15, 8))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title(f"{strings[0]} - Short-Time Fourier Transform")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    The main function of the script. It loads, preprocesses, and plots the EMG
    data from the specified files.
    """
    # load data
    extra_data = []
    total_data = pd.DataFrame()
    path = Path(__file__).parent / "data"
    for file in path.glob("*.txt"):
        if "uma" not in file.stem:
            continue  # only process correct files
        print(f"Processing file: {file}")
        stats, markers, data = load_data(file)
        extra_data.append(pd.DataFrame(stats.iloc[1, :]).T)
        data = preprocess_data(data)
        total_data = pd.concat([total_data, data], ignore_index=True)

    # calculate sampling frequency (hh:mm:ss)
    length_1 = extra_data[0].iloc[-1].values[-1]
    length_2 = extra_data[1].iloc[-1].values[-1]
    print(f"Length of file 1: {length_1}")
    print(f"Length of file 2: {length_2}")
    time_pattern = r"(\d{2}):(\d{2}):(\d{2})"
    match_1 = re.match(time_pattern, length_1)
    match_2 = re.match(time_pattern, length_2)
    if match_1 and match_2:
        hours_1, minutes_1, seconds_1 = map(int, match_1.groups())
        hours_2, minutes_2, seconds_2 = map(int, match_2.groups())
        total_seconds_1 = hours_1 * 3600 + minutes_1 * 60 + seconds_1
        total_seconds_2 = hours_2 * 3600 + minutes_2 * 60 + seconds_2
        total_time = total_seconds_1 + total_seconds_2
        total_samples = len(total_data)
        fs = total_samples / total_time
        print(f"Sampling frequency: {fs:.2f} Hz")
    else:
        print("Error: Time format is incorrect. Expected format is hh:mm:ss.")
        return

    # plot data
    plot_data(
        total_data.iloc[:, 1],
        total_data.iloc[:, 2],
        ["Total Data"],
        fs,
        window_size=15)
    stft_plot(
        total_data.iloc[:, 1],
        fs,
        ["Total Data - Channel 1"])
    stft_plot(
        total_data.iloc[:, 2],
        fs,
        ["Total Data - Channel 2"])


def test():
    """
    The main function of the script. It loads, preprocesses, and plots the EMG
    data from the specified files.
    """
    # load data
    extra_data = []
    total_data = pd.DataFrame()
    path = Path(__file__).parent / "data"
    for file in path.glob("*.txt"):
        print(f"Processing file: {file}")
        if "uma" in file.stem:
            print(f"Skipping file: {file}")
            continue  # process all files except correct ones
        stats, markers, data = load_data(file)
        extra_data.append(pd.DataFrame(stats.iloc[1, :]).T)
        data = preprocess_data(data)
        total_data = pd.concat([total_data, data], ignore_index=True)

    # calculate sampling frequency (hh:mm:ss)
    length_1 = extra_data[0].iloc[-1].values[-1]
    print(f"Length of file 1: {length_1}")
    time_pattern = r"(\d{2}):(\d{2}):(\d{2})"
    match_1 = re.match(time_pattern, length_1)
    if match_1 :
        hours_1, minutes_1, seconds_1 = map(int, match_1.groups())
        total_time = hours_1 * 3600 + minutes_1 * 60 + seconds_1
        total_samples = len(total_data)
        fs = total_samples / total_time
        print(f"Sampling frequency: {fs:.2f} Hz")
    else:
        print("Error: Time format is incorrect. Expected format is hh:mm:ss.")
        return

    # plot data
    plot_data(
        total_data.iloc[:, 1],
        total_data.iloc[:, 2],
        ["Total Data"],
        fs,
        window_size=15)
    stft_plot(
        total_data.iloc[:, 1],
        fs,
        ["Total Data - Channel 1"])
    stft_plot(
        total_data.iloc[:, 2],
        fs,
        ["Total Data - Channel 2"])


if __name__ == "__main__":
    main()
    # test()
