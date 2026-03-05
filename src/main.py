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
        window_size: int = 25,
        verbose: bool = False
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

    rms_1 = np.sqrt((data_1**2).rolling(window=window_size).mean())
    rms_2 = np.sqrt((data_2**2).rolling(window=window_size).mean())

    time = np.arange(len(data_1)) / fs

    if verbose is True:
        moving_avg_1 = data_1.rolling(window=window_size).mean()
        moving_avg_2 = data_2.rolling(window=window_size).mean()
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))
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
    else:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        axs[0].plot(time, data_1,
                    label="channel 1 (raw)")
        axs[0].plot(time, data_2,
                    label="channel 2 (raw)")
        axs[1].plot(time[:len(rms_1)], rms_1,
                    label="channel 1 (RMS)")
        axs[1].plot(time[:len(rms_2)], rms_2,
                    label="channel 2 (RMS)")
        axs[0].set_title(f"{strings[0]} - Raw Data")
        axs[1].set_title(f"{strings[0]} - RMS ({window_size})")

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


def get_marked_data(
        data: pd.DataFrame,
        marker: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Extracts the EMG data corresponding to specific markers.

    Parameters
    ----------
    data : pd.DataFrame
        The EMG data from which to extract the marked segment.
        (part 1 or part two, not total data)
    marker : pd.DataFrame
        Marker indicating the segment to be extracted from the EMG data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the data corresponding to the specified marker.
    """

    # helper function to find starting index and ending index of the marker
    def find_index(
            data_loc: pd.DataFrame,
            timestamp: float
            ) -> int:
        """
        Finds the index in the EMG data corresponding to a given timestamp.

        Parameters
        ----------
        data : pd.DataFrame
            The EMG data in which to find the index.
        timestamp : float
            The timestamp for which to find the corresponding index (seconds).

        Returns
        -------
        int
            The index in the EMG data corresponding to the given timestamp.
        """
        # find closest nearest indices
        closest_indices: list[int] = []
        for i in range(len(data_loc)):
            candidate = data_loc.iloc[i, 0]
            if not isinstance(candidate, str):
                continue
            match = re.match(r"(\d+)h: (\d+)m: (\d+)sec", candidate)
            if match:
                hours, minutes, seconds = map(int, match.groups())
                total_seconds = hours * 3600 + minutes * 60 + seconds
                if timestamp >= total_seconds >= timestamp-1:
                    closest_indices.append(i)
                if timestamp+1 >= total_seconds >= timestamp:
                    closest_indices.append(i)
                    break

        if len(closest_indices) == 1:
            closest_indices.append(len(data_loc) - 1)

        if len(closest_indices) == 1:
            closest_indices.append(len(data_loc) - 1)

        i_range: int = closest_indices[1] - closest_indices[0]
        index = int(i_range * (timestamp - timestamp // 1))
        return index + closest_indices[0]

    def parse_timestamp(
            timestamp: str
            ) -> float:
        """
        Parses a timestamp string in the format "hh:mm:ss" and converts it to
        total seconds.

        Parameters
        ----------
        timestamp : str
            The timestamp string to be parsed.

        Returns
        -------
        float
            The total number of seconds represented by the timestamp.
        """
        time_pattern = r"(\d{2}):(\d{2}):(\d{2}).(\d{7})"
        match = re.match(time_pattern, timestamp)
        if match:
            hours, minutes, seconds, mili = map(int, match.groups())
            total_seconds = hours * 3600 + minutes * 60 + seconds + mili / 1e7
            return total_seconds
        else:
            raise ValueError(f"Error: Time format is incorrect: {timestamp}")

    start_time = marker.iloc[0, 2]
    start_time = parse_timestamp(start_time)
    start_index = find_index(data, start_time)
    duration = marker.iloc[0, 3]
    duration = parse_timestamp(duration)
    end_time = start_time + duration
    end_index = find_index(data, end_time)

    # print(f"{start_time=}\t {end_time=}\t\t{start_index=}\t {end_index=}")

    return data.iloc[start_index:end_index, :]


def plot_mvc_normalization(
        base_ch1: pd.Series,
        mvc_ch1: pd.Series,
        ex_ch1: pd.Series,
        base_ch2: pd.Series,
        mvc_ch2: pd.Series,
        ex_ch2: pd.Series,
        strings: list[str],
        fs: float,
        window_size: int = 25
        ) -> None:
    """
    Plots MVC-normalized RMS signal (Channel 1 and 2).

    Baseline mean -> 0%
    MVC mean -> 100%
    Exercise shown after Vaseline and MVC.
    """

    def compute_rms(signal: pd.Series) -> pd.Series:
        return np.sqrt((signal ** 2).rolling(window=window_size).mean())

    def trimmed_signal(signal: pd.Series,
                       trim_start: float = 0.25,
                       trim_end: float = 0.05
                       ) -> float:
        n = len(signal)
        start = int(n * trim_start)
        end = int(n * (1 - trim_end))
        return signal.iloc[start:end]

    mvc_ch1 = trimmed_signal(mvc_ch1)
    mvc_ch2 = trimmed_signal(mvc_ch2)

    rms_base_1 = compute_rms(base_ch1)
    rms_mvc_1 = compute_rms(mvc_ch1)
    rms_ex_1 = compute_rms(ex_ch1)

    rms_base_2 = compute_rms(base_ch2)
    rms_mvc_2 = compute_rms(mvc_ch2)
    rms_ex_2 = compute_rms(ex_ch2)

    mean_base_1 = rms_base_1.mean()
    mean_mvc_1 = rms_mvc_1.mean()

    mean_base_2 = rms_base_2.mean()
    mean_mvc_2 = rms_mvc_2.mean()

    def normalize(rms, mean_base, mean_mvc):
        return (rms - mean_base) / (mean_mvc - mean_base) * 100

    norm_base_1 = normalize(rms_base_1, mean_base_1, mean_mvc_1)
    norm_mvc_1 = normalize(rms_mvc_1, mean_base_1, mean_mvc_1)
    norm_ex_1 = normalize(rms_ex_1, mean_base_1, mean_mvc_1)

    norm_base_2 = normalize(rms_base_2, mean_base_2, mean_mvc_2)
    norm_mvc_2 = normalize(rms_mvc_2, mean_base_2, mean_mvc_2)
    norm_ex_2 = normalize(rms_ex_2, mean_base_2, mean_mvc_2)

    t_base_1 = np.arange(len(norm_base_1)) / fs
    t_mvc_1 = np.arange(len(norm_mvc_1)) / fs

    calib_duration = max(t_base_1[-1], t_mvc_1[-1])

    t_ex_1 = np.arange(len(norm_ex_1)) / fs + calib_duration

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    axs[0].plot(t_base_1, norm_base_1, label="Baseline")
    axs[0].plot(t_mvc_1, norm_mvc_1, label="MVC")
    axs[0].plot(t_ex_1, norm_ex_1, label="Exercise")

    axs[0].axhline(0, linestyle="--", color="grey")
    axs[0].axhline(100, linestyle="--", color="grey")

    axs[0].axvline(calib_duration, linestyle="-", color="black")

    axs[0].set_title(f"{strings[0]} - Channel 1 MVC Normalized")
    axs[0].set_ylabel("Activation (% MVC)")
    axs[0].legend()

    t_base_2 = np.arange(len(norm_base_2)) / fs
    t_mvc_2 = np.arange(len(norm_mvc_2)) / fs

    calib_duration_2 = max(t_base_2[-1], t_mvc_2[-1])
    t_ex_2 = np.arange(len(norm_ex_2)) / fs + calib_duration_2

    axs[1].plot(t_base_2, norm_base_2, label="Baseline")
    axs[1].plot(t_mvc_2, norm_mvc_2, label="MVC")
    axs[1].plot(t_ex_2, norm_ex_2, label="Exercise")

    axs[1].axhline(0, linestyle="--", color="grey")
    axs[1].axhline(100, linestyle="--", color="grey")

    axs[1].axvline(calib_duration_2, linestyle="-", color="black")

    axs[1].set_title(f"{strings[0]} - Channel 2 MVC Normalized")
    axs[1].set_ylabel("Activation (% MVC)")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend()

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
    total_data_list = []
    path = Path(__file__).parent / "data"
    for file in path.glob("*.txt"):
        if "uma" not in file.stem:
            continue  # only process correct files
        print(f"Processing file: {file}")
        stats, markers, data = load_data(file)
        extra_data.append([
            pd.DataFrame(stats.iloc[1, :]).T,
            pd.DataFrame(markers)
            ])
        data = preprocess_data(data)
        total_data = pd.concat([total_data, data], ignore_index=True)
        total_data_list.append(data)

    if not total_data_list:
        print("No valid data files found.")
        return

    # calculate sampling frequency (hh:mm:ss)
    print(extra_data[0])
    length_1 = extra_data[0][0].iloc[-1].values[-1]
    length_2 = extra_data[1][0].iloc[-1].values[-1]
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

    names = ["baseline", "mvc", "ex1r", "ex1l", "ex2", "ex3", "ex4"]
    results = {}
    counter = 0
    for i, data in enumerate(total_data_list):
        for j in range(1, len(extra_data[i][1])):
            df = pd.DataFrame(extra_data[i][1].iloc[j, :]).T
            marked_data = get_marked_data(data, df)

            if counter < len(names):
                results[names[counter]] = marked_data
            counter += 1

    # plot data
    plot_data(
        total_data.iloc[:, 1],
        total_data.iloc[:, 2],
        ["Total Data"],
        fs,
        window_size=15,
        verbose=True)
    plot_data(
        results["baseline"].iloc[:, 1],
        results["baseline"].iloc[:, 2],
        ["baseline"],
        fs=fs,
        window_size=15)
    plot_mvc_normalization(
        results["baseline"].iloc[:, 1],
        results["mvc"].iloc[:, 1],
        results["ex4"].iloc[:, 1],

        results["baseline"].iloc[:, 2],
        results["mvc"].iloc[:, 2],
        results["ex4"].iloc[:, 2],

        ["MVC Normalization – Exercise 4"],
        fs,
        window_size=15
    )


def test():
    """
    Testing function of the script.
    Used only for testing, will change a lot.
    """
    pass


if __name__ == "__main__":
    main()
    # test()
