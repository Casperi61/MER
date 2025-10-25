import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import resample, hilbert, resample_poly
import pycochleagram.cochleagram as cgram
import warnings
import librosa

warnings.filterwarnings('ignore', message='divide by zero encountered')
warnings.filterwarnings('ignore', message='invalid value encountered')

# Folder path
folder_path = 'MEMD_audio'
mp3_files = glob.glob(os.path.join(folder_path, '*.mp3'))
print(f"Found {len(mp3_files)} MP3 files")

# Parameters
sr = 44100
n_freq_bands = 11
low_lim = 20
hi_lim = sr // 2
target_env_sr = 88


def custom_88hz_downsample(raw_envelopes, audio_sr, target_sr=88):
    """
    Custom downsample callable for pycochleagram.
    Receives: raw_envelopes (n_bands+2, time), audio_sr
    Returns: downsampled envelopes at target_sr Hz
    """
    n_bands_total, n_samples = raw_envelopes.shape

    # Calculate duration and target samples
    duration = n_samples / audio_sr
    target_samples = max(1, int(duration * target_sr))

    print(f"    Custom downsample: {n_samples} -> {target_samples} samples")

    # Resample each frequency band (including reconstruction filters)
    resampled = np.zeros((n_bands_total, target_samples))
    for i in range(n_bands_total):
        # Use resample for arbitrary ratios (handles non-integer perfectly)
        resampled[i, :] = resample(raw_envelopes[i, :], target_samples)

    return resampled


cochleagrams = []
coc = {}

for file_idx, file in enumerate(mp3_files):
    print(f"\nProcessing {file_idx + 1}/{len(mp3_files)}: {os.path.basename(file)}")

    try:
        # Load audio
        y, _ = librosa.load(file, sr=sr, mono=True)
        duration = len(y) / sr
        print(f"  Duration: {duration:.2f}s")

        # Create custom downsample function bound to target rate
        downsample_88hz = lambda envs: custom_88hz_downsample(envs, sr, target_env_sr)

        # Generate cochleagram with custom downsampling
        print("  Computing cochleagram with custom 88Hz downsampling...")
        coch = cgram.cochleagram(
            signal=y,
            sr=sr,
            n=n_freq_bands,
            low_lim=low_lim,
            hi_lim=hi_lim,
            sample_factor=1,
            downsample=downsample_88hz,  # Our custom callable!
            nonlinearity='db',
            ret_mode='envs',
            strict=False
        )

        print(f"  Full output shape: {coch.shape}")

        # Extract main 11 frequency bands (exclude reconstruction filters)
        main_bands = coch[1:1 + n_freq_bands, :]

        # main_bands = librosa.amplitude_to_db(np.maximum(main_bands, 1e-10), ref=np.max)

        print(f"  Main bands shape: {main_bands.shape}")

        # Transpose to (time, freq) as requested
        coch_time_freq = main_bands
        expected_frames = int(duration * target_env_sr)
        print(f"  Final shape: {coch_time_freq.shape} (expected ~{expected_frames})")

        # Verify time resolution
        actual_sr = coch_time_freq.shape[0] / duration
        print(f"  Actual time resolution: {actual_sr:.1f} Hz")
        coch_time_freq = np.array(coch_time_freq)[:, 88 * 15:88 * 45]
        cochleagrams.append(coch_time_freq)

        coc[file[file.find('/') + 1:-4]] = coch_time_freq

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        continue

print(f"\nSuccessfully processed {len(cochleagrams)}/{len(mp3_files)} files")
import pickle

# 保存数据
data = {'coc': coc}

with open('coc_data.pkl', 'wb') as f:
    pickle.dump(data, f)


# Alternative: Manual Gammatone Implementation (if pycochleagram still fails)
def manual_cochleagram_fallback(y, sr, n_bands=11, target_sr=88):
    """Simple manual implementation using scipy gammatone filters"""
    from scipy.signal import gammatone, hilbert
    from pycochleagram.erb import erbspace

    # ERB-spaced frequencies
    freqs = erbspace(low_lim, hi_lim, n_bands)

    # Generate filterbank (simplified)
    filters = []
    for f in freqs:
        # Gammatone filter parameters (simplified)
        b = 1.019 * 24.7 * (4.37 * f / 1000 + 1) ** 0.5  # ERB bandwidth
        filt = gammatone(N=64, f=f, bandwidth=b, fs=sr)
        filters.append(filt)

    # Apply filters and extract envelopes (this is computationally heavy)
    envelopes = []
    for filt in filters:
        filtered = np.convolve(y, filt, mode='same')
        env = np.abs(hilbert(filtered))
        envelopes.append(env)

    envelopes = np.array(envelopes)  # (n_bands, time)

    # Downsample to target rate
    duration = len(y) / sr
    target_samples = int(duration * target_sr)
    resampled = np.zeros((n_bands, target_samples))
    for i in range(n_bands):
        resampled[i, :] = resample_poly(envelopes[i, :], target_samples)

    return resampled.T  # (time, freq)


# Test fallback on first file if needed
if not cochleagrams and mp3_files:
    print("\n=== FALLBACK MODE: Manual Gammatone ===")
    try:
        y, _ = librosa.load(mp3_files[0], sr=sr, mono=True, duration=5.0)  # Test 5s
        fallback_coch = manual_cochleagram_fallback(y, sr, n_freq_bands, target_env_sr)
        print(f"Fallback success: {fallback_coch.shape}")

        # Quick plot
        plt.figure(figsize=(10, 4))
        plt.imshow(fallback_coch.T, aspect='auto', cmap='viridis')
        plt.title("Manual Gammatone Cochleagram (Fallback)")
        plt.xlabel('Time')
        plt.ylabel('Freq Bands')
        plt.colorbar()
        plt.show()

    except Exception as e:
        print(f"Fallback also failed: {e}")
