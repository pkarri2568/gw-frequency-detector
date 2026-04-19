import numpy as np
from edge_impulse_linux.runner import ImpulseRunner

class EIClassifier:
    def __init__(self, model_path):
        """
        Load the Edge Impulse .eim model and initialize the runner.
        """
        self.runner = ImpulseRunner(model_path)
        self.model_info = self.runner.init()
        print("Model initialized:", self.model_info)

    def classify(self, audio_float):
        """
        Classify a 1D numpy array of audio samples in float format (-1 to 1).
        Edge Impulse expects a flat list of int16 samples.
        """

        # Convert float audio (-1 to 1) → int16 (-32768 to 32767)
        audio_int16 = (audio_float * 32767).astype(np.int16)

        # Convert to Python list
        audio_list = audio_int16.tolist()

        # Run inference directly on the list
        result = self.runner.classify(audio_list)

        # Extract probabilities
        probs = result['result']['classification']

        return probs

