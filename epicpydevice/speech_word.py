from epiclibcpp.epiclib import (
    Symbol,
    geometric_utilities as gu,
    Speech_word as speech_word,
)

from datetime import datetime

from epicpydevice.geometric_utilities import Point

"""
Challenging to get structs to be recognized by EPICLib and still provide the
IDE inspection we're seeking for Python users. Current stategy is to create a class
with keyword only args. This will allow the use EPICLib C++ programmers are already
used to, namely:

sw = Speech_word()
sw.name = "whatever"

but it also allows Pythonistas to specify some or all of the vars in the initializer, e.g:

sw = Speech_word(name="watever)

level_left and level_right are still optional like in the C++ struct, but unlike there,
I had to move them to the end to satisfy Python.
"""


def timestamp() -> int:
    return int(round(datetime.now().timestamp()))


class Speech_word:
    """
    name: Symbol,  # unique name for each word object
    stream_name: Symbol,  # veridical stream name
    time_stamp: int,
    location: Point,
    pitch: float,  # pitch in semitones
    loudness: float,  # dB - level of physical, perceived loudness
    duration: float,
    content: Symbol,
    speaker_gender: Symbol,
    speaker_id: Symbol,
    # vvv These are not available in older versions of EPICpy
    utterance_id: int,  # utterance_id identifies the complete utterance in the corpus
    level_left: float = 0.0,
    level_right: float = 0.0
    """

    def __init__(
        self,
        *,
        name: Symbol,  # unique name for each word object
        stream_name: Symbol,  # veridical stream name
        time_stamp: int,
        location: gu.Point,
        pitch: float,  # pitch in semitones
        loudness: float,  # dB - level of physical, perceived loudness
        duration: float,
        content: Symbol,
        speaker_gender: Symbol,
        speaker_id: Symbol,
        utterance_id: int = timestamp(),  # id for the complete utterance in the corpus
        level_left: float = 0.0,
        level_right: float = 0.0
    ):
        ...

    # def __new__(cls, *args, **kwargs):
    def __new__(
        cls,
        name: Symbol,  # unique name for each word object
        stream_name: Symbol,  # veridical stream name
        time_stamp: int,
        location: gu.Point,
        pitch: float,  # pitch in semitones
        loudness: float,  # dB - level of physical, perceived loudness
        duration: float,
        content: Symbol,
        speaker_gender: Symbol,
        speaker_id: Symbol,
        utterance_id: int = timestamp(),  # id for the complete utterance in the corpus
        level_left: float = 0.0,
        level_right: float = 0.0,
    ):
        # return speech_word(*args, **kwargs)

        sw = speech_word()
        sw.name = name
        sw.stream_name = stream_name
        sw.time_stamp = time_stamp
        sw.location = location
        sw.pitch = pitch
        sw.loudness = loudness
        sw.duration = duration
        sw.level_left = level_left
        sw.level_right = level_right
        sw.content = content
        sw.speaker_gender = speaker_gender
        sw.speaker_id = speaker_id
        sw.utterance_id = utterance_id
        return sw


if __name__ == "__main__":
    s = Speech_word(
        name=Symbol("StimWord"),
        stream_name=Symbol("MainStream"),
        time_stamp=1000,
        location=Point(1, 1),
        pitch=1.0,
        loudness=1.0,
        duration=2.0,
        content=Symbol("MyWarning"),
        speaker_gender=Symbol("Male"),
        speaker_id=Symbol("TRO"),
        utterance_id=666,
    )
    print(s)
