import time
from pathlib import Path
from io import BytesIO
import base64
import csv
import re
from typing import Union, Any, List
import itertools

try:
    from ulid2 import generate_ulid_as_base32
except:
    import uuid
import pandas
from matplotlib.figure import Figure
from plum import dispatch

from epiclibcpp.epiclib import Output_tee
from epiclibcpp.epiclib import Device_base, Symbol, Speech_word
from epiclibcpp.epiclib.output_tee_globals import (Exception_out, Debug_out, Device_out)
import epiclibcpp.epiclib.geometric_utilities as gu

e_boxed_x = "\u274E"
e_boxed_check = "\u2611"


def unpack_param_string(
    pattern: str, delimiter: str = "|", left: str = "[", right: str = "]"
) -> List[str]:
    """
    Expand the brace-delimited possibilities in a string.
    E.g.: "10 Easy Dash" or "10 [Easy|Hard] Dash" or "10 [Easy|Hard] [Dash|HUD]"
    Based on solution from stackoverflow.com/MarekG Jan 2 22
    """
    seg_choices = (
        seg.strip(left + right).split(delimiter) if seg.startswith(left) else [seg]
        for seg in re.split(rf"(\{left}.*?\{right})", pattern)
    )

    return ["".join(parts) for parts in itertools.product(*seg_choices)]


class EpicPyDevice(Device_base):
    def __init__(
        self, parent: Any, ot: Output_tee, device_name: str, device_folder: Path
    ):
        # NOTE: ot is not being used, just use Device_out(...)
        super(EpicPyDevice, self).__init__(device_name, ot)

        self.parent = parent
        self.device_name = device_name
        self.device_folder = device_folder
        self.condition_string = ""

        self.rule_filename = ""  # will be adjusted automatically by simulation

        self.reparse_conditionstring = False

        # EPIC Simulation controller will know device is finished when
        # self.state == self.SHUTDOWN. You may want to manage state with enums,
        # so maybe states.START = 0, states.SHUTDOWN = 10, etc.
        self.state = 0
        self.SHUTDOWN = 1000

        # Dictionary, keys should be strings, values should be booleans
        # Any options you define here will be exposed to the user via a dialog window as
        # boolean checkboxes. This allows the user to alter some functionality of the
        # device from the EPIC gui during simulation runs.
        self.option = dict()

        # data
        # - Using a csv.writer() to ensure proper csv standards adherence!
        # - Use data.writerow(...) to save trial data (comma sep values in
        #                                              same order as header)
        self.data_filename = "data_output.csv"
        self.data_filepath = Path(
            self.device_folder, self.data_filename
        )  # *** REQUIRED BY EPICpy GUI!
        self.data_filemode = "a"  # append mode
        self.data_file = None
        self.data_writer = None
        self.data_header = ()

    """
    Methods Defined Here In EpicPyDevice
    """

    def stats_write(
        self, content: Union[str, Figure, pandas.DataFrame], *args, **kwargs
    ):
        """
        Device write method for objects meant for stats_window.
        Currently, accepts strings, matplotlib figures, and pandas dataframes
        This will be dynamically added to the device object after it has been
        loaded and instantiated.

        Warning: The Stats Output window is not cached! Writing large amounts
        of text (or small amounts very often) 《during》a simulation will cause
        significant slowdowns. This window is intended primarily for outputting
        statistical analyses 《after》a simulation has completed. "
        """

        if isinstance(content, str):
            color = kwargs["color"] if "color" in kwargs else ""
            text = content.replace("\n", "<br>")
            if color:
                text = f'<font color="{color}">{text}</font>'
        elif isinstance(content, Figure):
            # The figure will be converted to encoded text before going to output window.
            # https://stackoverflow.com/questions/48717794
            temp_file = BytesIO()
            content.savefig(temp_file, format="png")
            encoded = base64.b64encode(temp_file.getvalue()).decode("utf-8")
            text = f"<img src='data:image/png;base64,{encoded}'>"
        elif isinstance(content, pandas.DataFrame):
            text = content.to_html()
        elif isinstance(content, (int, float, list, tuple, dict)):
            return str(content)
        else:
            try:
                text = content
            except Exception as e:
                text = f"ERROR: epicpy_device_base:stats_write({type(content)}): {e}"

        self.parent.stats_win.write(text)

    def get_param_list(self) -> list:
        # the device is not the place to deal with ranged condition strings. If somehow
        # we've got one, just use the first permutation. Permutations are dealt with
        # elsewhere and thus this function should only be provided with a single
        # permutation. Just in case, the code below will strip off any range markers and
        # just take the first permutation
        param_set = unpack_param_string(self.condition_string)
        param_string = param_set[0]
        params = param_string.split(" ")
        return params

    def set_parameter_string(self, condition_string: str):
        self.condition_string = condition_string
        self.reparse_conditionstring = True

    def get_parameter_string(self):
        return self.condition_string

    def show_view_background(self, view_type: str, file_name: str, scaled: bool = True):
        """
        Sets background image for specified view.
        If file_name is empty, view image will be removed.
        file_name should be just the image file name.
        show_view_background assumes device file is next to folder called images
        and that the file specified by file_name is located in that folder.
        """
        try:
            assert view_type in (
                "visual",
                "auditory",
            ), "Incorrect view_type, should be in ('visual', 'auditory')"
            bg_file = Path(self.device_folder, "images", Path(file_name).name)
            self.parent.set_view_background_image(
                view_type=view_type, img_filename=str(bg_file), scaled=scaled
            )
        except AssertionError as e:
            Device_out(
                f"{e_boxed_x} ERROR in display_background parameter specification: {e}"
            )
        except FileNotFoundError as e:
            Device_out(
                f"{e_boxed_x} ERROR: Unable to locate view background image(s): [{e}]"
            )

    def delete_data_file(self):
        # ===============================================================================
        # >>> THIS METHOD IS EXPECTED BY THE GUI, BUT WON'T CRASH IF IT ISN'T PRESENT <<<
        # ===============================================================================

        self.finalize_data_output()

        try:
            self.data_filepath.unlink()
            Device_out(
                f'{e_boxed_check} Device "{self.device_name}" successfully deleted '
                f'data output file "{str(self.data_filepath)}"'
            )
        except Exception as e:
            Device_out(
                f'{e_boxed_check} Device "{self.device_name}" reported an error while '
                f'attempting to delete data output file "{str(self.data_filepath)}": '
                f"[{e}]"
            )

        self.init_data_output()

    def data_file_info(self) -> str:
        # ================================================================================
        # >>> THIS METHOD IS EXPECTED BY THE GUI, BUT WON'T CRASH IF IT ISN'T PRESENT <<<
        # ================================================================================

        # extra cautious, just fail gracefully if something goes wrong or
        #  data_filepath is null
        try:
            assert self.data_filepath.is_file()
            file_size = self.data_filepath.stat().st_size
            rows = len(self.data_filepath.read_text().splitlines())
            return f"Data Info: {rows - 1 if rows else 0} rows ({file_size} bytes)"
        except:
            return f"Data Info: ???"

    def init_data_output(self):
        # don't know if data file is opened or not, just try to close it anyway prior
        # to reopening it
        self.finalize_data_output()

        # try to open data file, don't stop on fail, just warn via device_out
        try:
            self.data_file = open(self.data_filepath, self.data_filemode)
        except IOError as e:
            Device_out(
                f"\n{e_boxed_x} WARNING: Unable to open device datafile at "
                f"{str(self.data_filepath)} "
                f"[{e}]!\n"
            )
            self.data_file = None

        if self.data_file:
            try:
                self.data_writer = csv.writer(self.data_file)
                if self.data_filemode == "w" or self.data_filepath.stat().st_size == 0:
                    self.data_writer.writerow(self.data_header)
            except IOError as e:
                self.data_file.close()
                self.data_file = None
                self.data_writer = None
                Device_out(
                    f"\n{e_boxed_x} WARNING: Unable to write to device datafile at "
                    f"{str(self.data_filepath)} [{e}].\n"
                )

    def finalize_data_output(self):
        try:
            self.data_file.flush()
            self.data_file.close()
        except:  # broad on purpose!
            pass

        self.data_file = None
        self.data_writer = None

    @staticmethod
    def unique_id() -> str:
        """
        Returns 26 char unique char string no matter how fast you call it.
        Realistically, it's unique as long as you don't call it more than
        1.21e+24 times per second!
        https://github.com/mdipierro/ulid
        """
        try:
            return generate_ulid_as_base32()
        except:
            return uuid.uuid5(uuid.NAMESPACE_URL, str(time.time_ns())).hex

    def accept_event(self, *args, **kwargs):
        raise NotImplementedError(
            f"epicpy_device_base.accept_event was called with {args=} {kwargs=}"
        )

    """
    Methods Defined in Device_base.h

    Note: having these defined like this has a slight impact on run time, but keeps 
    device-coding Pythonistas from having to read and understand the C++ code!
    """

    # EVENT HANDLING INTERFACE CALLED BY DEVICE_PROCESSOR
    # THESE METHODS DO NOTHING IN THIS CLASS; OVERRIDE ONLY THOSE NECESSARY FOR DEVICE
    # --------------------------------------------------------------------------------

    def handle_Start_event(self):
        ...

    def handle_Stop_event(self):
        ...

    def handle_Report_event(self, duration: int):
        ...

    def handle_Delay_event(
        self,
        _type: Symbol,
        datum: Symbol,
        object_name: Symbol,
        property_name: Symbol,
        property_value: Symbol,
    ):
        ...

    def handle_Keystroke_event(self, key_name: Symbol):
        ...

    def handle_Type_In_event(self, type_in_string: Symbol):
        ...

    def handle_hold_event(self, button_name: Symbol):
        ...

    def handle_Hold_event(self, button_name: Symbol):
        ...

    def handle_Release_event(self, button_name: Symbol):
        ...

    def handle_Click_event(self, button_name: Symbol):
        ...

    def handle_Double_Click_event(self, button_name: Symbol):
        ...

    def handle_Point_event(self, target_name: Symbol):
        ...

    def handle_Ply_event(
        self,
        cursor_name: Symbol,
        target_name: Symbol,
        new_location: gu.Point,
        movement_vector: gu.Polar_vector,
    ):
        ...

    # def handle_Vocal_event(self, vocal_input: Symbol, duration: Optional[int] = None):
    #     ...

    @dispatch
    def handle_Vocal_event(self, vocal_input: Symbol):
        ...

    @dispatch
    def handle_Vocal_event(self, vocal_input: Symbol, duration: int):
        ...

    def handle_VisualFocusChange_event(self, object_name: Symbol):
        ...

    def handle_Eyemovement_Start_event(
        self, target_name: Symbol, new_location: gu.Point
    ):
        ...

    def handle_Eyemovement_End_event(self, target_name: Symbol, new_location: gu.Point):
        ...

    def handle_HLGet_event(
        self, props: List[Symbol], values: List[Symbol], tag: Symbol
    ):
        """default response is to simply echo the information back with a dummy name"""
        self.make_high_level_input_appear(Symbol("HLDummyObject"), props, values, tag)

    def handle_HLPut_event(self, props: List[Symbol], values: List[Symbol]):
        ...

    # SERVICES FOR THE DERIVED DEVICE_BASE CLASS
    # ------------------------------------------

    # def get_time(self) -> int:
    #     """get the current time"""
    #     return super(EpicPyDevice, self).get_time()

    # def get_trace(self) -> bool:
    #     """get the trace state of the associated Device_processor"""
    #     return super(EpicPyDevice, self).get_trace()
    #
    # def set_trace(self, flag: bool):
    #     """
    #     set the trace state of the associated Device_processor - to allow,
    #     e.g. Dummy device to trace by default
    #     """
    #     super(EpicPyDevice, self).set_trace(flag)

    # FUNCTIONS TO MANIPULATE THE DEVICE'S DISPLAY AND OUTPUT
    # -------------------------------------------------------

    @dispatch
    def make_visual_object_appear(self, object_name: Symbol):
        """Tell the simulated human we have a new visual object"""
        super(EpicPyDevice, self).make_visual_object_appear(object_name)

    @dispatch
    def make_visual_object_appear(
        self,
        object_name: Symbol,
        location: gu.Point,
        size: gu.Size,
    ):
        # Tell sim. human we have new visual object with specified location & size
        super(EpicPyDevice, self).make_visual_object_appear(object_name, location, size)

    def set_visual_object_location(self, object_name: Symbol, new_location: gu.Point):
        """Tell the simulated human that location of a visual object has changed"""
        super(EpicPyDevice, self).set_visual_object_location(object_name, new_location)

    def set_visual_object_size(self, object_name: Symbol, new_size: gu.Size):
        """Tell the simulated human that size of a visual object has changed"""
        super(EpicPyDevice, self).set_visual_object_size(object_name, new_size)

    @dispatch
    def set_visual_object_property(
        self, object_name: Symbol, property_name: Symbol, property_value: Symbol
    ):
        """Tell the simulated human we have a value for a property of a visual object"""
        super(EpicPyDevice, self).set_visual_object_property(
            object_name, property_name, property_value
        )

    @dispatch
    def set_visual_object_property(
        self,
        object_name: Symbol,
        property_name: Symbol,
        property_value: Union[int, float],
    ):
        """Tell the simulated human we have a value for a property of a visual object"""
        super(EpicPyDevice, self).set_visual_object_property(
            object_name, property_name, Symbol(str(property_value))
        )

    @dispatch
    def set_visual_object_property(
        self, object_name: Symbol, property_name: Symbol, property_value: str
    ):
        """Tell the simulated human we have a value for a property of a visual object"""
        super(EpicPyDevice, self).set_visual_object_property(
            object_name, property_name, Symbol(property_value)
        )

    def make_visual_object_disappear(self, object_name: Symbol):
        """Tell the simulated human that a visual object is gone"""
        super(EpicPyDevice, self).make_visual_object_disappear(object_name)

    def set_auditory_stream_location(self, name: Symbol, location: gu.Point):
        """A new auditory stream with location"""
        super(EpicPyDevice, self).set_auditory_stream_location(name, location)

    def set_auditory_stream_size(self, name: Symbol, size: gu.Size):
        """The size of an auditory stream has changed"""
        super(EpicPyDevice, self).set_auditory_stream_size(name, size)

    def set_auditory_stream_property(
        self, name: Symbol, propname: Symbol, propvalue: Symbol
    ):
        """A property of an auditory stream has changed"""
        super(EpicPyDevice, self).set_auditory_stream_property(
            name, propname, propvalue
        )

    def make_auditory_event(self, message: Symbol):
        """An auditory event with a "message" as a simple signal"""
        super(EpicPyDevice, self).make_auditory_event(message)

    def make_auditory_sound_event(
        self,
        name: Symbol,
        stream: Symbol,
        location: gu.Point,
        timbre: Symbol,
        loudness: float,
        duration: int,
        intrinsic_duration: int = 0,
    ):
        """
        An auditory event with lcation, timbre, stream, duration, and optional
        intrinsic duration
        """
        super(EpicPyDevice, self).make_auditory_sound_event(
            name, stream, location, timbre, loudness, duration, intrinsic_duration
        )

    def make_auditory_sound_start(
        self,
        name: Symbol,
        stream: Symbol,
        location: gu.Point,
        timbre: Symbol,
        loudness: float,
        intrinsic_duration: int,
    ):
        """
        A new auditory sound with location, timbre, stream, and intrinsic duration
        has started
        """
        super(EpicPyDevice, self).make_auditory_sound_start(
            name, stream, location, timbre, loudness, intrinsic_duration
        )

    def make_auditory_sound_stop(self, name: Symbol):
        """The auditory sound has stopped"""
        super(EpicPyDevice, self).make_auditory_sound_stop(name)

    def set_auditory_sound_property(
        self, name: Symbol, propname: Symbol, propvalue: Symbol
    ):
        """A property of the auditory sound has changed"""
        super(EpicPyDevice, self).set_auditory_sound_property(name, propname, propvalue)

    def make_auditory_speech_event(self, word: Speech_word):
        super(EpicPyDevice, self).make_auditory_speech_event(word)

    def make_high_level_input_appear(
        self,
        object_name: Symbol,
        props: List[Symbol],
        values: List[Symbol],
        tag: Symbol,
    ):
        """
        Tell the simulated human that a High-Level input object with properties and
        values should be stored under a WM tag
        """
        super(EpicPyDevice, self).make_high_level_input_appear(
            object_name, props, values, tag
        )

    def make_high_level_input_disappear(self, object_name: Symbol):
        """Tell the simulated human that a High-Level input object is gone"""
        super(EpicPyDevice, self).make_high_level_input_disappear(object_name)

    @dispatch
    def schedule_delay_event(self, delay: int):
        """
        Create a device delay event with the specified contents, to arrive at the
        current time + the delay. The result is a call to the handle_Delay_event
        function at that time.
        """
        super(EpicPyDevice, self).schedule_delay_event(delay)

    @dispatch
    def schedule_delay_event(self, delay: int, delay_type: Symbol, delay_datum: Symbol):
        super(EpicPyDevice, self).schedule_delay_event(delay, delay_type, delay_datum)

    @dispatch
    def schedule_delay_event(
        self,
        delay: int,
        delay_type: Symbol,
        object_name: Symbol,
        property_name: Symbol,
        property_value: Symbol,
    ):
        raise NotImplementedError('ok, it is, but i am trying to see if this is every used!')
        super(EpicPyDevice, self).schedule_delay_event(
            delay, delay_type, object_name, property_name, property_value
        )

    def make_report(self, time: int, duration: int):
        """
        Tell the simulated human that a report with the specified duration must be made
        """
        super(EpicPyDevice, self).make_report(time, duration)

    def set_human_parameter(self, proc_name: str, param_name: str, spec: str):
        """Allow the device to set a human parameter - for generating parametric runs"""
        super(EpicPyDevice, self).set_human_parameter(proc_name, param_name, spec)

    def get_human_prs_filename(self) -> str:
        """
        Allow the device to access the human's production rule set filename for
        documentation purposes
        """
        return super(EpicPyDevice, self).get_human_prs_filename()

    # AVAILABLE BUT FORBIDDEN IN EPICPY

    def stop_simulation(self):
        # Device_out("ERROR: Do Not Use Device_base.stop_simulation() with EPICpy!")
        # raise NotImplementedError(
        #     "Do Not Use Device_base.stop_simulation() with EPICpy!"
        # )
        return super(EpicPyDevice, self).stop_simulation()

    def __getattr__(self, name):
        def _missing(*args, **kwargs):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("A missing method was called.")
            print(f"The object was {self}, the method was {name}. ")
            print(f"It was called with {args} and {kwargs} as arguments\n")

        return _missing
