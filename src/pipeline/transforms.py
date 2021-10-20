"""Transforms for audio pipeline"""

from fastai.data.transforms import Transform, Pipeline

from src.data.audio import AudioObject, AudioArray, SpecArray, SpecTensor


# pylint: disable=method-hidden
# pylint: disable=no-self-use
class Tensorify(Transform):
    """Turn spectrogram to tensor object"""
    def encodes(self, spec: SpecArray):
        """Encodes spectrogram to tensor"""
        return spec.to_tensor()

    def decodes(self, spec: SpecTensor):
        """Decodes spectrogram to array"""
        return spec.to_array()


# pylint: disable=super-init-not-called
class AudioProcessor(Transform):
    """Turn spectrogram to tensor object"""
    def __init__(self, sr=22050, duration=5):
        self.sr = sr
        self.duration = duration

    def encodes(self, audio: AudioObject):
        """Encodes spectrogram to tensor"""
        audio.sr = self.sr
        audio.clip(self.duration)
        return audio


PoiPipeline = Pipeline([AudioArray.from_file, AudioProcessor])
