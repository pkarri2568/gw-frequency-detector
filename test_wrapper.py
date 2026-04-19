import numpy as np
from inference_wrapper import EIClassifier

# Load model
clf = EIClassifier("/Users/pranny/Desktop/IR_Product/mission_model.eim")

# Create a fake audio sample (1 second of silence)
fs = 16000
audio = np.zeros(fs, dtype=np.float32)

# Classify
probs = clf.classify(audio)

print("Probabilities:", probs)
