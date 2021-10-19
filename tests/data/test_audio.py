"""Test for audio data types"""

import unittest

import librosa
import numpy as np
from fastai.torch_basics import Tensor

from src.data.audio import AudioArray, AudioTensor

class TestAudioArray(unittest.TestCase):
    """Test AudioArray object type definition"""
    fn = "data/AudioTest1.wav"
    def test_init(self):
        """Test loading file with librosa and init"""
        sig, sr = librosa.load(self.fn)
        audio_array = AudioArray(sig, sr, np.array)
        assert_audio_array(audio_array)
    
    def test_from_file(self):
        """Test from_file object initialization"""
        audio_array = AudioArray.from_file(self.fn)
        assert_audio_array(audio_array)
    
    def test_to_tensor(self):
        """Test transforming array to tensor"""
        audio_array = AudioArray.from_file(self.fn)
        audio_tensor = audio_array.to_tensor()


class TestAudioTensor(unittest.TestCase):
    """Test AudioTensor object type definition"""
    fn = "data/AudioTest1.wav"
    def test_init(self):
        """Test loading file with librosa and init"""
        sig, sr = librosa.load(self.fn)
        audio_tensor = AudioTensor(sig, sr, Tensor)
        assert_audio_tensor(audio_tensor)
    
    def test_from_file(self):
        """Test from_file object initialization"""
        audio_tensor = AudioTensor.from_file(self.fn)
        assert_audio_tensor(audio_tensor)

    def test_to_array(self):
        """Test transforming tensor to array"""
        audio_tensor = AudioTensor.from_file(self.fn)
        audio_array = audio_tensor.to_array()
        assert_audio_array(audio_array)

def assert_audio_array(audio_array):
    assert isinstance(audio_array, AudioArray)
    assert isinstance(audio_array.sig, np.ndarray)
    assert isinstance(audio_array.sr, int)
    assert audio_array.sr == 22050
    assert isinstance(audio_array.duration, float)
    assert audio_array.duration == 4.115056689342404

def assert_audio_tensor(audio_tensor):
    assert isinstance(audio_tensor, AudioTensor)
    assert isinstance(audio_tensor.sig, Tensor)
    assert isinstance(audio_tensor.sr, int)
    assert audio_tensor.sr == 22050
    assert isinstance(audio_tensor.duration, float)
    assert audio_tensor.duration == 4.115056689342404
