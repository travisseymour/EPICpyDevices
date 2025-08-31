from typing import Any

from epicpydevice.epicpy_visual_encoder_base import EPICPyVisualEncoder
from epicpydevice.symbol import Symbol
from epicpydevice.random_utilities import unit_uniform_random_variable
from epicpydevice.standard_utility_symbols import Nil_c
from epicpydevice.standard_symbols import (
    Text_c,
    Color_c,
    Red_c,
    Green_c,
    Blue_c,
    Yellow_c,
    Shape_c,
    Circle_c,
    Filled_Rectangle_c,
)


# EpicPy will expect all visual encoders to be of class
# VisualEncoder and subclassed from EPICPyVisualEncoder.


class VisualEncoder(EPICPyVisualEncoder):
    def __init__(self, encoder_name: str, parent: Any = None):
        super(VisualEncoder, self).__init__(
            encoder_name=encoder_name if encoder_name else "VisualEncoder",
            parent=parent,
        )
        self.recoding_failure_rate = 0.5

    def set_object_property(
        self,
        object_name: Symbol,
        property_name: Symbol,
        property_value: Symbol,
        encoding_time: int,
    ) -> bool:
        """
        Imparts a self.recoding_failure_rate chance of mis-perceiving object colors:
        Yellow->Blue & Green->Red.
        If not overridden, this method just returns False to indicate that the encoding
        is not being handled here.
        """

        # this encoding does not apply
        if property_name != Color_c:
            return False

        # failure rate is 0, nothing to do
        if not self.recoding_failure_rate:
            return False

        # can only confuse green and yellow
        if property_value not in (Yellow_c, Green_c, Nil_c):
            return False

        # flip a coin to decide whether encoding is successful
        successful_encoding = (
            unit_uniform_random_variable() > self.recoding_failure_rate
        )

        if property_value == Nil_c:
            # previous property values need to be removed!
            encoding = Nil_c
        elif property_value == Yellow_c:
            # random chance of perceiving yellow as blue!
            encoding = property_value if successful_encoding else Blue_c
        elif property_value == Green_c:
            # random chance of perceiving yellow as red!
            encoding = property_value if successful_encoding else Red_c
        else:
            return False  # this encoding does not apply

        # transmit forward the encoded Shape
        self.schedule_change_property_event(
            encoding_time, object_name, property_name, encoding
        )

        # this encoding applied
        return True
