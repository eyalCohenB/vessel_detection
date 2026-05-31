from random import random
import pickle
import numpy as np
from scipy.signal import firwin, lfilter, hilbert
import soundfile as sf
import matplotlib.pyplot as plt
import os


def plot_demon_matrix(DemonMat, title="Demon Matrix"):
    DemonMat = np.asarray(DemonMat)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        DemonMat,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )
    plt.colorbar(label="Normalized value")
    plt.xlabel("Demon bin")
    plt.ylabel("Block index")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def matlab_envelope_default(x: np.ndarray) -> np.ndarray:
    """
    Match MATLAB envelope(x) default analytic behavior as documented:
    - remove mean
    - compute analytic signal magnitude
    - add mean back

    MATLAB doc:
    envelope(x) returns envelopes as magnitude of analytic signal,
    initially removes the mean and adds it back after computing.
    """
    x = np.asarray(x)

    # Preserve MATLAB-like behavior for real-valued 1-D input
    x_mean = np.mean(x)
    x_centered = x - x_mean

    analytic = hilbert(x_centered)
    y_upper = np.abs(analytic) + x_mean

    return y_upper


def demon(y, Fs, fpass, fstop, fpassDemon):
    """
    Direct translation of MATLAB:

    function [SNR,demon]=demonRoee(y,Fs, fpass, fstop, fpassDemon)
    """
    y = np.asarray(y).squeeze()

    # %% filter parameter
    # n=1024;
    n = 1024

    # MATLAB:
    # b=fir1(n,[fpass fstop]/Fs*2);
    #
    # fir1 order n => n+1 taps
    # normalized cutoff [fpass fstop]/(Fs/2)
    # in SciPy, passing fs=Fs with raw Hz cutoffs is equivalent.
    b = firwin(
        numtaps=n + 1,
        cutoff=[fpass, fstop],
        window="hamming",
        pass_zero="bandpass",
        scale=True,
        fs=Fs,
    )

    # %% Bandpass filter 3-10k
    # y1=filter(b,1,y);%filter the signal
    y1 = lfilter(b, [1.0], y)

    # %% ****Roee
    # Faxis = linspace(-Fs/2, Fs/2, length(y1));
    Faxis = np.linspace(-Fs / 2.0, Fs / 2.0, len(y1))

    # pos = find(Faxis > 2 & Faxis < fpassDemon);
    pos = np.where((Faxis > 2) & (Faxis < fpassDemon))[0]

    # y2 = abs(fftshift(fft(envelope(y1))));
    env_y1 = matlab_envelope_default(y1)
    y2 = np.abs(np.fft.fftshift(np.fft.fft(env_y1)))

    # demon=y2(pos);
    demon = y2[pos]

    # M = max(demon);
    M = np.max(demon)

    # N= mean(demon);
    N = np.mean(demon)

    # SNR = 10*log10(M/N);
    SNR = 10.0 * np.log10(M / N)

    # demon=demon./max(demon);
    demon = demon / np.max(demon)

    return SNR, demon


def buildDemonMat(sig, fs, BlockLen, fpass, fstop, fpassDemon):
    """
    Direct translation of MATLAB:

    function [DemonMat, SNRVec] = buildDemonMat(sig, Fs, BlockLen, fpass, fstop, fpassDemon)
    """
    sig = np.asarray(sig).squeeze()

    # Duration = length(sig) / Fs;
    Duration = len(sig) / fs

    # NumBlocks = floor(Duration / BlockLen);
    NumBlocks = int(np.floor(Duration / BlockLen))

    # LenDemon = Fs/2 * BlockLen / (Fs/2) * (fpassDemon - 2);
    #
    # Keep the same formula structure as MATLAB.
    # Algebraically this is BlockLen * (fpassDemon - 2).
    # MATLAB uses this as a size, so cast to int for NumPy allocation.
    LenDemon = int(fs / 2 * BlockLen / (fs / 2) * (fpassDemon - 2))

    # SNRVec = zeros(1, NumBlocks);
    SNRVec = np.zeros(NumBlocks, dtype=float)

    # DemonMat = zeros(NumBlocks, LenDemon);
    DemonMat = np.zeros((NumBlocks, LenDemon), dtype=float)

    # for BlockInd = 1: NumBlocks
    for BlockInd in range(1, NumBlocks + 1):

        # CurrentSig = sig((BlockInd-1)*BlockLen*Fs+1: BlockInd*BlockLen*Fs);
        #
        # MATLAB is 1-based and inclusive on the right.
        # Python is 0-based and exclusive on the right.
        start_idx = int((BlockInd - 1) * BlockLen * fs)
        end_idx = int(BlockInd * BlockLen * fs)
        CurrentSig = sig[start_idx:end_idx]

        # [SNR, DemonRes] = demonRoee(CurrentSig, Fs, fpass, fstop, fpassDemon);
        SNR, DemonRes = demon(CurrentSig, fs, fpass, fstop, fpassDemon)

        # SNRVec(BlockInd) = SNR;
        SNRVec[BlockInd - 1] = SNR

        # DemonMat(BlockInd, :) = DemonRes';
        DemonMat[BlockInd - 1, :] = DemonRes

    return DemonMat, SNRVec



def main():
    base_path = '../data/talmon_data'
    data_type = ["train", "test"]

    for dt in data_type:
        print(f"Processing {dt} data...")
        files =os.listdir(f"{base_path}/{dt}_data")
        files = [f for f in files if f.endswith(".wav") or f.endswith(".flac")]

        skip_files = os.listdir(f"{base_path}/{dt}_demon")
        # shuffle files and pick the first 10 to test
        # np.random.shuffle(files)
        # files = files[:10]
        files.sort()  # sort files for consistent processing order

        for i in range(len(files)):
            if files[i].replace(".wav", ".pkl").replace(".flac", ".pkl") in skip_files:
                print(f"Skipping {files[i]} as it has already been processed.")
                continue
            print(f"\nProcessing file {i+1}/{len(files)}: {files[i]}")
            data, fs = sf.read(f"{base_path}/{dt}_data/{files[i]}")
            sig = data[:, 0] if data.ndim > 1 else data

            DemonMat, SNRVec = buildDemonMat(sig=sig,fs=fs,BlockLen=1,fpass=100,fstop=1500,fpassDemon=300)


            print("DemonMat shape:", DemonMat.shape)
            print(f"DemonMat value range: [{DemonMat.min():.4f}, {DemonMat.max():.4f}]")
            print("SNRVec shape:", SNRVec.shape)

            
            # save DemonMat and SNRVec to a pickle file for later inspection
            demon_file_name = os.path.splitext(files[i])[0] + ".pkl"
            print(f"Saving DemonMat and SNRVec to {base_path}/{dt}_demon/{demon_file_name}")
            with open(f"{base_path}/{dt}_demon/{demon_file_name}", "wb") as f:
                pickle.dump({"DemonMat": DemonMat, "SNRVec": SNRVec}, f)

            # plot_demon_matrix(DemonMat)

    
if __name__ == "__main__":
    main()