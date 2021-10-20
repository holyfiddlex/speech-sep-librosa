"""Error definitions for system"""

class AudioDurationMismatchError(Exception):
    """Error for when audio duration is not what was expected"""

class AudioTypeMismatchError(Exception):
    """Error when audio type is not what was expected"""

class AudioSampleRateMismatchError(Exception):
    """Error when audio sample rate is not what was expected"""

class SpecShapeTooBigError(Exception):
    """Error when trimming spec shape is too big"""
