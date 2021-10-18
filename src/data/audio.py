"""Audio and Spectrogram definitions"""

import numpy as np
import librosa


class Audio():
    """Audio object definition. Used as a superclass for AudioTensor and AudioArray"""
    def __init__(self, sig, sr, filename=None):
        """Initializes Audio object"""
        self.sig = sig
        self.sr = sr
        self.filename = filename

    @classmethod
    def from_file(cls, fn: str, data_type = None):
        """Creates Audio object from filename with the specified data type."""
        sig, sr = librosa.load(fn)
        audio = cls(data_type(sig), sr, fn)
        return audio

    def show(self):
        """Plots time series of audio"""
        raise NotImplementedError

    def listen(self):
        """In jupyter enables reproducer to listen to audio"""
        raise NotImplementedError

    def to_spec(self):
        """Transforms Audio object to Spec object of the same data type"""
        raise NotImplementedError

    def resample(self, sr: int):
        """Resamples time series using specified sample rate"""
        raise NotImplementedError

    def clip(self, time: float):
        """Clip audio to specified amount of time in seconds"""
        raise NotImplementedError

class Spectrogram():
    """Spectrogram object definition. Used as a superclass for SpecTensor and SpecArray"""
    def __init__(self):
        """Initializes Spectrogram object"""
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename: str, data_type: str):
        """Creates Audio object from filename with the specified data type."""
        raise NotImplementedError

    def show(self):
        """Plots spectrogram image"""
        raise NotImplementedError

    def to_audio(self):
        """Transforms Spectrogram object to Audio of the same data type"""
        raise NotImplementedError

    def trim(self):
        """Trim 2d shape to fit U-Net model"""
        raise NotImplementedError

class AudioArray(Audio):
    """Audio object with numpy array"""
    @classmethod
    def from_file(cls, fn, data_type=np.array):
        super().__class__(sig, sr, data_type)
