"""Transforms for audio pipeline"""

from fastai.data.transforms import (
    ItemTransform,
    Transform,
    Pipeline,
    ToTensor,
)

from src.data.audio import (
    AudioObject,
    AudioArray,
    SpecArray,
    SpecTensor,
    SpecObject,
)

from src.data.mask import MaskArray, MaskTensor

from src.utils.errors import (
    AudioDurationMismatchError,
    AudioTypeMismatchError,
    AudioSampleRateMismatchError,
)


# pylint: disable=method-hidden
# pylint: disable=no-self-use
class Tensorify(Transform):
    """Turn spectrogram to tensor object"""
    def encodes(self, spec: SpecArray):
        """Encodes spectrogram to tensor"""
        return spec.to_tensor()
    
    def encodes(self, mask: MaskArray):
        """Encodes mask to tensor"""
        return mask.to_tensor()

    def decodes(self, spec: SpecTensor):
        """Decodes spectrogram to array"""
        return spec.to_array()


class Spectify(Transform):
    """Turn spectrogram to tensor object"""
    def encodes(self, audio: AudioObject):
        """Encodes audio to spectrogram"""
        return audio.to_spec()

    def decodes(self, spec: SpecObject):
        """Decodes spectrogram to audio"""
        return spec.to_array()


# pylint: disable=super-init-not-called
class MaskifyIBM(ItemTransform):
    """Turn two spectrograms to mask object"""
    def __init__(self, umbral: float):
        self.umbral = umbral

    def encodes(self, specs):
        """Creates mask from spectrograms"""
        mask = MaskTensor((specs[0].data > self.umbral)*1)
        return tuple([specs[-1], mask])


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


class AudioMixer(ItemTransform):
    """Turn spectrogram to tensor object"""

    def encodes(self, audios):
        """Joins audios together"""
        if len(set([len(audio) for audio in audios])) != 1:
            raise AudioDurationMismatchError("Recieved audios of different lengths")
        if len(set([audio.sr for audio in audios])) != 1:
            raise AudioSampleRateMismatchError("Recieved audios of different sample rates")
        if len(set([type(audio) for audio in audios])) != 1:
            raise AudioTypeMismatchError("Recieved audios of different type")

        mixed_signal = sum([audio.sig for audio in audios])
        mixed_name = " with ".join([str(audio.fn) for audio in audios])
        mixed_audio = type(audios[0])(mixed_signal, audios[0].sr, "mixed audios: "+mixed_name)
        return tuple([*audios, mixed_audio])


class SpecTimmer(Transform):
    """Trim spectrogram to fit into model"""
    def __init__(self, shape: tuple):
        self.shape = shape

    def encodes(self, spec: SpecObject):
        """Trims spectrogram into specified shape"""
        spec.trim(self.shape)
        return spec


class DebugPrinter(Transform):
    """Simple transform used to debug pipeline"""
    def encodes(self, o):
        """Prints returns object"""
        print(o)
        return o


PoiPipeline = Pipeline([
    AudioArray.from_file,
    AudioProcessor(),
    AudioMixer(),
    Spectify(),
    SpecTimmer((1024, 176)),
    MaskifyIBM(0.1),
    Tensorify(),
    ToTensor(),
])
