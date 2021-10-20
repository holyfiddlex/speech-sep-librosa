"""Audio and Spectrogram definitions"""

import math
import librosa
import matplotlib.pyplot as plt

import numpy as np
from fastai.torch_basics import Tensor

from scipy.signal import resample_poly
from IPython.display import Audio, display


class AudioObject():
    """Audio object definition. Used as a superclass for AudioTensor and AudioArray"""

    data_type: callable = None
    color_axis: int = 2

    # pylint: disable=not-callable
    def __init__(self, sig, sr, fn=None):
        """Initializes Audio object"""
        self.sig = self.data_type(sig)
        self.__sr = sr
        self.fn = fn

    @classmethod
    def from_file(cls, fn: str):
        """Creates Audio object from filename with the specified data type."""
        try:
            sig, sr = librosa.load(fn)
            return cls(sig, sr, fn)
        except TypeError as err:
            raise NotImplementedError("Error, function was not implemented") from err

    @property
    def duration(self):
        """Returns the duration of the signal in seconds"""
        return len(self.sig)/self.sr

    @property
    def sr(self):
        """Manages samplerate changes and resamples accordingly"""
        return self.__sr

    @sr.setter
    def sr(self, sr):
        self.__resample(sr)
        self.__sr = sr

    def __resample(self, sr: int):
        """Resamples time series using specified sample rate"""
        if self.sr == sr:
            return
        sr_gcd = math.gcd(self.sr, sr)
        resampled = resample_poly(self.sig, int(sr/sr_gcd), int(self.sr/sr_gcd), axis=-1)
        self.sig = resampled

    def show(self):
        """Plots time series of audio"""
        _, ax = plt.subplots()
        ax.plot(self.sig)
        return ax

    def listen(self):
        """In jupyter enables reproducer to listen to audio"""
        display(Audio(self.sig, rate=self.sr))

    def clip(self, time: float):
        """Clip audio to specified amount of time in seconds"""
        if time >= self.duration:
            #TODO fill in this part of the code
            pass
        else:
            self.sig = self.sig[:time*self.sr]

    def to_spec(self):
        """Transforms Audio object to Spec object of the same data type"""


class AudioArray(AudioObject):
    """Audio object with numpy array"""

    data_type: callable = np.array

    def to_tensor(self):
        """Returns Tensor version of object"""
        return AudioTensor(self.sig, self.sr, self.fn)

    def to_spec(self):
        """Transforms audio object to spectrogram and concatenates real and imag parts"""
        spec = librosa.stft(self.sig)
        spec3d = np.stack([abs(spec), spec.real, spec.imag], axis=self.color_axis)
        return SpecArray(spec3d, self.sr, self.fn)


class AudioTensor(AudioObject):
    """Audio object with numpy array"""

    data_type: callable = Tensor

    def to_array(self):
        """Returns np.array version of object"""
        return AudioArray(self.sig, self.sr, self.fn)

    def to_spec(self):
        """Transforms audio object to spectrogram and concatenates real and imag parts"""
        spec = librosa.stft(np.array(self.sig))
        spec3d = np.stack([abs(spec), spec.real, spec.imag], axis=self.color_axis)
        return SpecTensor(Tensor(spec3d), self.sr, self.fn)


class SpecObject():
    """Spectrogram object definition. Used as a superclass for SpecTensor and SpecArray"""

    data_type: callable = None
    audio_type: type = None

    # pylint: disable=not-callable
    def __init__(self, data, sr, fn=None):
        """Initializes Spectrogram object"""
        self.data = self.data_type(data)
        self.sr = sr
        self.fn = fn

    @classmethod
    def from_file(cls, fn: str):
        """Creates Audio object from filename with the specified data type."""
        audio = cls.audio_type.from_file(fn)
        return audio.to_spec()

    @property
    def shape(self):
        """Return shape of spec data"""
        return self.data.shape

    def show(self):
        """Plots spectrogram image"""
        _, ax = plt.subplots()
        return ax.pcolormesh(self.data[:,:,0])

    def to_audio(self):
        """Transforms Spectrogram object to Audio of the same data type"""
        real = np.array(self.data[:,:,1])
        imag = np.array(self.data[:,:,2])
        sig = librosa.istft(real+imag*1j)
        return self.audio_type(sig, self.sr, self.fn)

    # def trim(self):
    #     """Trim 2d shape to fit U-Net model"""
    #     raise NotImplementedError


class SpecArray(SpecObject):
    """Spectrogram for array objects"""
    data_type: callable = np.array
    audio_type: type = AudioArray

    def to_tensor(self):
        """Returns SpecArray data as SpecTensor"""
        return SpecTensor(self.data, self.sr, self.fn)


class SpecTensor(SpecObject):
    """Spectrogram for tensor objects"""
    data_type: callable = Tensor
    audio_type: type = AudioTensor

    def to_array(self):
        """Returns SpecTensor data as SpecArray"""
        return SpecArray(self.data, self.sr, self.fn)
