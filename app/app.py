import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import io

from inference_wrapper import EIClassifier


# Load Edge Impulse model

@st.cache_resource
def load_models():
    mission_model = EIClassifier("/Users/pranny/Desktop/IR_Product/gw_freq_edgeimpulse/model/mission_model.eim")
    naive_model = EIClassifier("/Users/pranny/Desktop/IR_Product/gw_freq_edgeimpulse/model/naive_model.eim")
    return mission_model, naive_model

mission_clf, naive_clf = load_models()


# Audio generation utilities

FS = 16000
DURATION = 1.0

def generate_tone(freq, duration=DURATION, fs=FS):
    """
    Generate a clean sine wave tone.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    return t, tone

def add_noise(signal, noise_level=0.1):
    """
    Add Gaussian noise.
    """
    noise = np.random.normal(0, noise_level, size=len(signal))
    return signal + noise

def add_frequency_drift(signal, freq, drift_strength=0.0, fs=FS):
    """
    Add slow frequency drift across the signal.
    Drift_strength = fraction of freq to drift by.
    """
    if drift_strength == 0:
        return signal

    n = len(signal)
    t = np.linspace(0, 1, n)
    drift = freq * drift_strength * t
    phase = 2 * np.pi * (freq * t + drift * t)
    return np.sin(phase)

def add_glitches(signal, glitch_prob=0.0, glitch_strength=0.5):
    """
    Add short bursts of noise (glitches).
    """
    if glitch_prob == 0:
        return signal

    sig = signal.copy()
    n = len(sig)
    for i in range(n):
        if np.random.rand() < glitch_prob:
            sig[i] += glitch_strength * np.random.randn()
    return sig

def add_gaps(signal, gap_prob=0.0, gap_length=200):
    """
    Randomly zero out short segments (gaps).
    """
    if gap_prob == 0:
        return signal

    sig = signal.copy()
    n = len(sig)
    for i in range(n):
        if np.random.rand() < gap_prob:
            start = i
            end = min(i + gap_length, n)
            sig[start:end] = 0
    return sig

def normalize_audio(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal



# Streamlit UI

st.title("Hybrid Frequency-Band Detector")

st.markdown("""
This app simulates a simplified gravitational-wave detection pipeline using control sliders where:
- A tone is generated at a chosen frequency  
- Noise, drift, glitches, and gaps simulate mission-aware conditions  
- An Edge Impulse model classifies the tone  
- A classical rule checks if the frequency is in the signal band  
- Hybrid verification combines ML + classical  
""")

# Sidebar controls
st.sidebar.header("Controls")

freq = st.sidebar.slider("Frequency (Hz)", 100, 1000, 500, 10)
noise_level = st.sidebar.slider("Noise level", 0.0, 0.5, 0.1, 0.01)
drift_strength = st.sidebar.slider("Frequency drift strength", 0.0, 0.2, 0.0, 0.01)
glitch_prob = st.sidebar.slider("Glitch probability", 0.0, 0.05, 0.0, 0.001)
gap_prob = st.sidebar.slider("Gap probability", 0.0, 0.05, 0.0, 0.001)

threshold = st.sidebar.slider("ML detection threshold", 0.5, 0.99, 0.8, 0.01)
use_hybrid = st.sidebar.checkbox("Use hybrid verification", value=True)


# Single-sample classification

if st.button("Generate tone and classify"):
    # Generate clean tone
    t, tone = generate_tone(freq)

    # Apply drift
    tone = add_frequency_drift(tone, freq, drift_strength=drift_strength)

    # Add noise
    noisy_tone = add_noise(tone, noise_level=noise_level)

    # Add glitches
    noisy_tone = add_glitches(noisy_tone, glitch_prob=glitch_prob)

    # Add gaps
    noisy_tone = add_gaps(noisy_tone, gap_prob=gap_prob)

    # Normalize audio before sending to model
    noisy_tone = normalize_audio(noisy_tone)

    # Plot waveform
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, noisy_tone)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Noisy tone at {freq} Hz")
    st.pyplot(fig)

    # Download audio
    buf = io.BytesIO()
    sf.write(buf, noisy_tone, FS, format='WAV')
    st.download_button(
        label="Download audio",
        data=buf.getvalue(),
        file_name=f"tone_{freq}Hz.wav",
        mime="audio/wav"
    )

    # Run both classifiers
    mission_probs = mission_clf.classify(noisy_tone)
    naive_probs = naive_clf.classify(noisy_tone)

    # Extract probabilities
    p_signal_mission = mission_probs.get('signal_band', 0.0)
    p_non_signal_mission = mission_probs.get('non_signal_band', 0.0)

    p_signal_naive = naive_probs.get('naive_signal_band', 0.0)
    p_non_signal_naive = naive_probs.get('naive_non_signal_band', 0.0)

    # Model output (mission-aware)
    st.subheader("Mission-Aware Model Output")
    st.write(f"Signal band probability: {p_signal_mission:.3f}")
    st.write(f"Non-signal band probability: {p_non_signal_mission:.3f}")

    # Model output (naive)
    st.subheader("Naive Model Output")
    st.write(f"Signal band probability: {p_signal_naive:.3f}")
    st.write(f"Non-signal band probability: {p_non_signal_naive:.3f}")

    # Detection decisions
    ml_detected_mission = p_signal_mission >= threshold
    ml_detected_naive = p_signal_naive >= threshold

    st.subheader("Detection Decisions")
    st.write(f"Mission-aware ML: {'DETECTED' if ml_detected_mission else 'NOT DETECTED'}")
    st.write(f"Naive ML: {'DETECTED' if ml_detected_naive else 'NOT DETECTED'}")

    # Classical decision
    classical_detected = (400 <= freq <= 600)

    # Hybrid decision (mission-aware model only)
    if use_hybrid:
        final_detected = ml_detected_mission and classical_detected
        mode_desc = "Hybrid"
    else:
        final_detected = ml_detected_mission
        mode_desc = "ML-only"

    st.write(f"Classical decision: {'DETECTED' if classical_detected else 'NOT DETECTED'}")
    st.write(f"Final mission-aware decision ({mode_desc}): {'DETECTED' if final_detected else 'NOT DETECTED'}")


# Robustness testing

st.markdown("---")
st.subheader("Robustness Test (Multiple Random Tones)")

n_test = st.number_input("Number of random test samples", min_value=10, max_value=500, value=50, step=10)

if st.button("Run robustness test"):
    # Initialize counters
    mission_correct = 0
    naive_correct = 0
    classical_correct = 0
    hybrid_correct = 0

    mission_false_alarms = 0
    naive_false_alarms = 0
    classical_false_alarms = 0
    hybrid_false_alarms = 0

    for _ in range(int(n_test)):
        # Randomly choose signal or non-signal
        is_signal = np.random.rand() < 0.5

        if is_signal:
            freq_rand = np.random.uniform(400, 600)
        else:
            if np.random.rand() < 0.5:
                freq_rand = np.random.uniform(100, 300)
            else:
                freq_rand = np.random.uniform(700, 1000)

        # Generate tone
        t, tone = generate_tone(freq_rand)
        tone = add_frequency_drift(tone, freq_rand, drift_strength=drift_strength)
        noisy_tone = add_noise(tone, noise_level=noise_level)
        noisy_tone = add_glitches(noisy_tone, glitch_prob=glitch_prob)
        noisy_tone = add_gaps(noisy_tone, gap_prob=gap_prob)

        # Run both models
        mission_probs = mission_clf.classify(noisy_tone)
        naive_probs = naive_clf.classify(noisy_tone)

        p_signal_mission = mission_probs.get('signal_band', 0.0)
        p_signal_naive = naive_probs.get('signal_band', 0.0)

        # ML decisions
        mission_det = p_signal_mission >= threshold
        naive_det = p_signal_naive >= threshold

        # Classical decision
        classical_det = (400 <= freq_rand <= 600)

        # Hybrid decision (mission-aware ML + classical)
        hybrid_det = mission_det and classical_det

        # Accuracy tracking
        mission_correct += (mission_det == is_signal)
        naive_correct += (naive_det == is_signal)
        classical_correct += (classical_det == is_signal)
        hybrid_correct += (hybrid_det == is_signal)

        # False alarms (only when true label = non-signal)
        if not is_signal:
            if mission_det:
                mission_false_alarms += 1
            if naive_det:
                naive_false_alarms += 1
            if classical_det:
                classical_false_alarms += 1
            if hybrid_det:
                hybrid_false_alarms += 1

    # Display results
    st.write("### Model Accuracy")
    st.write(f"Mission-aware ML accuracy: {mission_correct / n_test:.2f}")
    st.write(f"Naive ML accuracy: {naive_correct / n_test:.2f}")
    st.write(f"Classical accuracy: {classical_correct / n_test:.2f}")
    st.write(f"Hybrid accuracy: {hybrid_correct / n_test:.2f}")
