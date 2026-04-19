import os
import numpy as np
import soundfile as sf

# Sampling rate for audio (Edge Impulse often uses 16 kHz)
FS = 16000
DURATION = 1.0  # seconds

def generate_tone(freq, duration=DURATION, fs=FS):
    """
    Generate a pure sine wave tone at a given frequency.
    - freq: frequency in Hz
    - duration: length of the tone in seconds
    - fs: sampling rate (samples per second)
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    return tone

def add_noise(signal, noise_level=0.1):
    """
    Add Gaussian noise to the signal.
    - noise_level: standard deviation of the noise
    """
    noise = np.random.normal(0, noise_level, size=len(signal))
    return signal + noise

def save_wav(filename, mission_data, fs=FS):
    """
    Save a 1D numpy array as a WAV file.
    """
    sf.write(filename, mission_data, fs)

def main():
    # Paths
    signal_dir = os.path.join("mission_data", "mission_signal_band")
    nonsignal_dir = os.path.join("mission_data", "mission_non_signal_band")
    os.makedirs(signal_dir, exist_ok=True)
    os.makedirs(nonsignal_dir, exist_ok=True)

    # Define frequency ranges
    # Signal band: 400–600 Hz (toy "GW-like" band)
    signal_min, signal_max = 400, 600
    # Non-signal band: 100–300 Hz and 700–1000 Hz
    nonsignal_ranges = [(100, 300), (700, 1000)]

    n_samples_per_class = 300  # can increase

    # Generate signal band samples
    for i in range(n_samples_per_class):
        freq = np.random.uniform(signal_min, signal_max)
        tone = generate_tone(freq)
        # Add random noise level to simulate mission-aware training
        noisy_tone = add_noise(tone, noise_level=np.random.uniform(0.05, 0.2))
        filename = os.path.join(signal_dir, f"signal_{i}_freq_{int(freq)}.wav")
        save_wav(filename, noisy_tone)
        print("Saved", filename)

    # Generate non-signal band samples
    for i in range(n_samples_per_class):
        # Randomly choose one of the non-signal ranges
        r = nonsignal_ranges[np.random.randint(0, len(nonsignal_ranges))]
        freq = np.random.uniform(r[0], r[1])
        tone = generate_tone(freq)
        noisy_tone = add_noise(tone, noise_level=np.random.uniform(0.05, 0.2))
        filename = os.path.join(nonsignal_dir, f"nonsignal_{i}_freq_{int(freq)}.wav")
        save_wav(filename, noisy_tone)
        print("Saved", filename)

if __name__ == "__main__":
    main()
