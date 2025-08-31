from typing import Any

from epicpy.epiclib.epiclib import Auditory_encoder_base, Symbol
from epicpy.epiclib.epiclib.standard_utility_symbols import Nil_c
from epicpy.epiclib.epiclib import geometric_utilities as gu
from epicpy.epiclib.epiclib.output_tee_globals import (Normal_out, Exception_out, Debug_out)

# EpicPy will expect all auditory encoders to be of class
# AuditoryEncoder and subclassed from Auditory_encoder_base
# OR
# AuditoryEncoder and subclassed from EPICPyAuditoryEncoder.


class EPICPyAuditoryEncoder(Auditory_encoder_base):
    def __init__(self, encoder_name: str, parent: Any = None):
        super(EPICPyAuditoryEncoder, self).__init__(
            encoder_name if encoder_name else "AuditoryEncoder"
        )
        self.parent = parent

        self.recoding_failure_rate = 0.0

    def set_object_property(
        self,
        object_name: Symbol,
        property_name: Symbol,
        property_value: Symbol,
        encoding_time: int,
    ) -> bool:
        # if this encoding does not apply, return False
        if property_name == Nil_c:
            return False

        # do something
        encoded_property = property_value  # default, does nothing

        # transmit forward the encoded property
        self.schedule_change_property_event(
            encoding_time, object_name, property_name, encoded_property
        )

        # return true so EPIC knows that this encoding has been handled
        return True

    def recode_location(self, original_location: gu.Point) -> gu.Point:
        # default is to just return the orig location
        return original_location

    def handle_Delay_event(
        self, objet_name: Symbol, property_name: Symbol, property_value: Symbol
    ) -> bool:
        # default is to return False indicting this event not being handled here
        return False
