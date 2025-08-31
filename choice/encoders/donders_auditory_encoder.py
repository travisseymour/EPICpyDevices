from epicpydevice.epicpy_auditory_encoder_base import EPICPyAuditoryEncoder
import epicpydevice.geometric_utilities as GU
from epicpydevice.random_utilities import unit_uniform_random_variable

# EpicPy will expect all auditory encoders to be of class
# AuditoryEncoder and subclassed from EPICPyAuditoryEncoder.


class AuditoryEncoder(EPICPyAuditoryEncoder):
    def __init__(self, encoder_name: str, parent):
        super(AuditoryEncoder, self).__init__(
            encoder_name=encoder_name if encoder_name else "AuditoryEncoder",
            parent=parent,
        )

        self.recoding_failure_rate = 0.5

    def recode_location(self, original_location: GU.Point) -> GU.Point:
        """
        Imparts a self.recoding_failure_rate of mis-perceiving the y location
        of an auditory stimulus. If not overridden, default will return original_location
        """

        x, y = original_location.x, original_location.y

        # Currently, in the choice and detection devices, all auditory stimuli are
        # either at (-5, 5) or (5, 5) let's have a self.recoding_failure_rate
        # chance of perceiving the y location as 0

        # flip a coin to decide whether encoding is successful
        successful_encoding = (
            unit_uniform_random_variable() > self.recoding_failure_rate
        )

        y = y if successful_encoding else 0

        return GU.Point(x, y)
