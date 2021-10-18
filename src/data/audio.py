"""Audio and Spectrogram definitions"""

class Audio():
    """Audio object definition. Used as a superclass to AudioTensor and AudioArray"""
    def __init__(self):
        """Initializes Audio object"""

    @classmethod
    def from_file(cls, filename: str, data_type: str):
        """Creates Audio object from filename with the specified data type."""

    def show(self):
        """Plots time series of audio"""

    def listen(self):
        """In jupyter enables reproducer to listen to audio"""

    def to_spec(self):
        """Transforms Audio object to Spec object of the same data type"""

    # pylint: disable=invalid-name
    def resample(self, sr: int):
        """Resamples time series using specified sample rate"""
