from textwrap import dedent

from pathlib import Path
from enum import Enum, auto
import random

from datetime import datetime
from typing import List, Optional, Any

from epicpydevice import epicpy_device_base
from epicpydevice.symbol import Symbol
from epicpydevice.output_tee import Output_tee
from epicpydevice.output_tee_globals import (Device_out, Exception_out, Debug_out)
from epicpydevice.symbol_utilities import concatenate_to_Symbol
from epicpydevice.device_exception import Device_exception
from epicpydevice.speech_word import Speech_word

import epicpydevice.geometric_utilities as GU
from epicpydevice.standard_symbols import Red_c, Green_c, Color_c
from epicpydevice.standard_symbols import (
    Shape_c,
    Circle_c,
    Cross_c,
    Square_c,
    Triangle_c,
    Cross_Hairs_c,
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# without this, figures may cut off some axis labels
rcParams.update({"figure.autolayout": True})


class state(Enum):
    START = auto()
    START_TRIAL = auto()
    REMOVE_FIXATION_POINT = auto()
    PRESENT_PROBE = auto()
    PRESENT_PROBE2 = auto()
    PRESENT_SEARCH_OBJECTS = auto()
    WAITING_FOR_RESPONSE = auto()
    FINISH_TRIAL = auto()
    SHUTDOWN = auto()


class Distance(Enum):
    NEAR = 0
    FAR = 1


class MatchType(Enum):
    BOTH = 0
    COLOR = 1
    SHAPE = 2
    NEITHER = 3


class SearchObject:
    def __init__(
        self,
        _id: int,
        location: GU.Point,
        color: Symbol,
        shape: Symbol,
        name: Symbol = None,
    ):
        self._id: int = _id  # an ID number - turned into object name
        self.location: GU.Point = location
        self.color: Symbol = color
        self.shape: Symbol = shape
        self.name: Symbol = name if name else Symbol()

    def __repr__(self) -> str:
        return (
            f"SearchObject(id: {self._id}, location: {self.location}, "
            f"color: {self.color}, shape: {self.shape}, name: {self.name}"
        )


# handle internal constants
Fixation_point_c = Symbol("Fixation_point")
Display_c = Symbol("Display")
Loudspeaker_c = Symbol("Loudspeaker")


# EpicPy will expect all devices to be of class "EpicDevice" and subclassed from
# "EpicPyDevice" or "epicpy_device.EpicPyDevice"


class EpicDevice(epicpy_device_base.EpicPyDevice):
    def __init__(self, ot: Output_tee, parent: Any, device_folder: Path):
        epicpy_device_base.EpicPyDevice.__init__(
            self,
            parent=parent,
            ot=ot,
            device_name="Findlay1997_Device_v2022.3",
            device_folder=device_folder,
        )

        # NOTE: ot is not being used, just use Device_out(...)
        self.device_name = "Findlay1997_Device_v2022.3"
        self.condition_string = "100"  # from EpicPyDevice, default is ''

        self.task_name = "Findlay1997"

        self.trial = 0

        self.reparse_conditionstring = False  # from EpicPyDevice, default is False

        # EPIC Simulation controller will know device is finished when
        # self.state == self.SHUTDOWN
        self.state = state.START
        self.SHUTDOWN = state.SHUTDOWN

        # Optionally expose some boolean device options to the user via the GUI.
        self.option = dict()  # from EpicPyDevice, default is dict()
        # useful for showing debug information during development
        self.option["show_debug_messages"] = False
        # useful for showing device state info during trial run
        self.option["show_trial_events"] = False
        # useful for outputting trial parameters and data during task run
        self.option["show_trial_data"] = True
        # useful for long description of the task and parameters
        self.option["show_task_description"] = True
        # useful for timing runs
        self.option["show_task_statistics"] = True

        # a datafile is automatically managed by EpicPyDevice.
        # Use self.data_writer.writerow(TUPLE OF VALUES) to write each row of csv data.
        # Need to first define the data header or your csv file could be malformed:
        self.data_header = (
            "TaskName",
            "Trial",
            "EML",
            "DistType",
            "CorrectResponse",
            "Response",
            "Mode",
            "RT",
            "ACC",
            "MatchType",
            "Device",
            "RuleFile",
            "Timestamp",
        )

        # parameters
        self.n_objects = 16
        self.n_trials = 10

        self.colors = [Red_c, Green_c]
        self.shapes = [Circle_c, Cross_c, Square_c, Triangle_c]
        self.target_shapes = [Circle_c, Cross_c]

        # stimulus generation
        self.search_objects: List[SearchObject] = list()
        self.locations: List[GU.Point] = list()
        self.setup_locations()
        try:
            assert len(self.locations) == self.n_objects
        except AssertionError as e:
            Device_out(f"ERROR: Unable to initialize device: {e}")
            raise

        # display constants
        self.search_object_size = GU.Size(1.2, 1.2)
        self.probe_location = GU.Point(0.0, 0.0)

        # display state
        self.probe: Optional[SearchObject] = None
        self.target_num = 0
        self.target: Optional[SearchObject] = None
        self.distance_index = 0  # 0 for near, 1 for far

        # count the number of time we have a target of each characteristic
        # color, shape, distance
        self.n_target_type = list()

        # assumes device file is next to folder called images!
        # self.show_view_background(
        #     view_type='visual', file_name='visual.jpg', scaled=True
        # )
        # self.show_view_background(
        #     view_type='auditory', file_name='auditory.jpg', scaled=True
        # )

        if self.option["show_task_description"]:
            self.describe_task()

    def setup_locations(self):
        # generate the two circles by calculation, using angles 0 - 315.
        angles = [GU.to_radians(i * 45.0) for i in range(8)]
        sizes = [5.7, 10.2]
        location_array = [[0] * 8, [0] * 8]
        for i in range(2):
            for j in range(8):
                # FIXME: need to bring in the overloaded operators from geometry class or this stuff will never work right!!
                location_array[i][j] = GU.Point(0.0, 0.0) + GU.Polar_vector(
                    sizes[i], angles[j]
                )

        # print these and distance from
        for i in range(2):
            # Device_out('\n')
            for j in range(8):
                target_distance = GU.cartesian_distance(
                    location_array[i][j], GU.Point(0.0, 0.0)
                )

        # fill a vector with all of them
        for i in range(2):
            for j in range(8):
                self.locations.append(location_array[i][j])

        # calculate the average distances of each point from all of the others
        # total up reciprocals
        for i in range(len(self.locations)):
            dist = 0.0
            total_inv_dist = 0.0
            for j in range(len(self.locations)):
                if j == i:
                    continue
                d = GU.cartesian_distance(self.locations[i], self.locations[j])
                dist += d
                total_inv_dist += 1 / d
            dist = dist / (len(self.locations) - 1)

    def parse_condition_string(self):
        error_msg = f"{self.condition_string}\n Should be: number of trials(int > 0)"

        # returns items from space delimited condition string as strings.
        # Uses self.get_param_list() instead of self.condition_string.split(' ')
        # to ensure any accidentally remaining range information is ignored.
        params = self.get_param_list()

        if not params or not params[0].isdigit():
            raise Device_exception(
                f"Error: Incorrect condition string: {error_msg}, "
                f"Should just contain the number of trials, e.g.: '400'.",
            )

        nt = int(params[0]) if params[0].isdigit() else 0
        if nt > 0:
            self.n_trials = nt
        else:
            raise Device_exception(
                f"Error: Number of trials must be positive, not [{nt}]."
            )

        self.reparse_conditionstring = False

    def initialize(self):
        """
        Initializes run. Called whenever model is stopped and started anew.
        I.e., NOT called, when model is paused and resumed.
        """
        self.state = state.START

        self.search_objects.clear()

        self.trial = 0

        # self.n_target_type = [[[0, 0] for _ in range(2)] for _ in range(3)]  # [2][2][2]
        self.n_target_type = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]

        # from EpicPyDevice, recommended for your initialize method!
        self.init_data_output()

    def describe_task(self):
        text = f"""
        **********************************************************************
        Device Name: {self.device_name}
        **********************************************************************
        A demonstration device based on the Findlay 1997 Exp. 5 task.

        For each trial,
            1. Put up a fixation point for a delay, wait a delay, 
            2. Put up a probe object.
            3. Wait for a delay.
            4. Put up several objects at their locations, one of which is the same as the probe.
            5. Wait for response (say Z for incorrect response, / for correct response)
            7. record score, RT
            
        The objects have a certain size, and appear on in two circles, with certain constraints.
        Each object has an individual name.
        **********************************************************************
        """
        Device_out(dedent(text))

    def handle_Start_event(self):
        """
        You have to get the ball rolling with a first time-delayed event -
        nothing happens until you do.
        """
        self.schedule_delay_event(500)

    def handle_Stop_event(self):
        """
        called after the stop_simulation function (which is part of the base device class)
        """
        if self.device_out:
            Device_out(f"{self.processor_info()} received Stop_event\n")
        self.finalize_data_output()
        self.refresh_experiment()

    def handle_Delay_event(
        self,
        _type: Symbol,
        datum: Symbol,
        object_name: Symbol,
        property_name: Symbol,
        property_value: Symbol,
    ):
        global DEVICE_FINISHED
        if self.state == state.START:
            self.state = state.START_TRIAL
            self.start_trial()
            self.schedule_delay_event(1000, Symbol("StartTrial"), Symbol())

        elif self.state == state.START_TRIAL and _type == Symbol("StartTrial"):
            self.present_fixation_point()
            self.state = state.REMOVE_FIXATION_POINT
            # make trial start time fluctuate
            self.schedule_delay_event(
                400 + random.randrange(0, 100), Symbol("RemoveFixation"), Symbol()
            )

        elif self.state == state.REMOVE_FIXATION_POINT and _type == Symbol(
            "RemoveFixation"
        ):
            self.remove_fixation_point()
            self.state = state.PRESENT_PROBE
            self.schedule_delay_event(100, Symbol("PresentProbe"), Symbol())

        elif self.state == state.PRESENT_PROBE and _type == Symbol("PresentProbe"):
            self.present_probe()
            self.state = state.PRESENT_PROBE2
            self.schedule_delay_event(150, Symbol("PresentProbe2"), Symbol())

        elif self.state == state.PRESENT_PROBE2 and _type == Symbol("PresentProbe2"):
            self.present_probe_word2()
            self.state = state.PRESENT_SEARCH_OBJECTS
            self.schedule_delay_event(1000, Symbol("PresentSearchObjects"), Symbol())

        elif self.state == state.PRESENT_SEARCH_OBJECTS and _type == Symbol(
            "PresentSearchObjects"
        ):
            self.present_search_objects()
            self.state = state.WAITING_FOR_RESPONSE
            self.schedule_delay_event(50, Symbol("WaitForResp"), Symbol())

        elif self.state == state.WAITING_FOR_RESPONSE and _type == Symbol(
            "WaitForResp"
        ):
            ...
        elif self.state == state.FINISH_TRIAL and _type == Symbol("FinishTrial"):
            self.setup_next_trial()
        elif self.state == state.SHUTDOWN:
            self.stop_simulation()  # doesn't do anything in EpicPy...could just omit
        elif _type == Symbol("Nil"):
            ...
        else:
            raise Device_exception(
                f"Device delay event is unknown or improper device state:\n"
                f"Event_State={self.state}. Event_Type={str(_type)}.",
            )

    def start_trial(self):
        if self.option["show_debug_messages"]:
            Device_out("*trial_start|")

        if self.reparse_conditionstring:
            self.parse_condition_string()
            self.initialize()

        self.trial += 1

        if self.option["show_debug_messages"]:
            Device_out("trial_start*\n")

    def present_fixation_point(self):
        if self.option["show_debug_messages"]:
            Device_out("*present_fixation|")

        self.make_visual_object_appear(
            Fixation_point_c, self.probe_location, GU.Size(1.0, 1.0)
        )
        self.set_visual_object_property(Fixation_point_c, Color_c, Red_c)
        self.set_visual_object_property(Fixation_point_c, Shape_c, Cross_Hairs_c)

        if self.option["show_debug_messages"]:
            Device_out("present_fixation*\n")

    def remove_fixation_point(self):
        if self.option["show_debug_messages"]:
            Device_out("*remove_fixation|")

        self.make_visual_object_disappear(Fixation_point_c)

        if self.option["show_debug_messages"]:
            Device_out("remove_fixation*\n")

    def present_probe(self):
        if self.option["show_debug_messages"]:
            Device_out("*present_probe_word_1|")

        # following does not match Findlay 1997x5 stimulus balancing in that colors
        # do not alternate, and shapes are equal in number - his description sounds
        # like some shapes may have appeared more often than not. However, there are a
        # total of three foils that have the same shape and the opposite color.
        # If colors alternate, that means there is an equal number of each color.

        # get a randomization of the possible locations
        random.shuffle(self.locations)

        self.search_objects.clear()

        # create a random probe object

        probe_color = random.randrange(0, len(self.colors))
        probe_shape = random.randrange(0, len(self.target_shapes))
        assert 0 <= probe_color < 2
        assert 0 <= probe_shape < 2

        other_color = 1 if probe_color == 0 else 0

        iobject = 0

        # put the target object in (1), save a copy of it - is not complete at this time
        # because only when we output it, do we create its full name
        self.target = SearchObject(
            _id=iobject,
            location=self.locations[iobject],
            color=self.colors[probe_color],
            shape=self.shapes[probe_shape],
            name=Symbol(),
        )
        self.search_objects.append(self.target)
        self.target_num = 0  # it is the first in the search_objects vector.
        iobject += 1

        # now make 3 objects with the same shape and the other color and put them in (3)
        for i in range(1, 4):
            self.search_objects.append(
                SearchObject(
                    iobject,
                    self.locations[iobject],
                    self.colors[other_color],
                    self.shapes[probe_shape],
                )
            )
            iobject += 1

        # at this point we have 1 of the probe color and 3 of the other color.
        # To balance number of color, we need 7 more of the probe color and
        # 5 more of the other color.
        color_sequence = [probe_color] * 7
        color_sequence.extend([other_color] * 5)
        random.shuffle(color_sequence)

        # for the remaining shapes, pick a randomized color,
        # and put in four of each shape (12)
        i_col_seq = 0

        for i, shape in enumerate(self.shapes):
            if i == probe_shape:
                continue
            for _ in range(4):
                icolor = color_sequence[i_col_seq]
                i_col_seq += 1
                self.search_objects.append(
                    SearchObject(
                        iobject, self.locations[iobject], self.colors[icolor], shape
                    )
                )
                iobject += 1

        assert iobject == self.n_objects

        # probe ID is the number of objects - one more than the search stimuli
        self.probe = SearchObject(
            self.n_objects,
            self.probe_location,
            self.colors[probe_color],
            self.target_shapes[probe_shape],
        )

        # classify whether the target was near or far
        target_distance = GU.cartesian_distance(
            self.target.location, self.probe.location
        )
        self.distance_index = (
            0 if target_distance < 8.0 else 1
        )  # check against Findlay's

        self.n_target_type[probe_color][probe_shape][self.distance_index] += 1

        probe_word = Speech_word(
            name=Symbol("ColorWord"),
            stream_name=Symbol("LeftSpeaker"),
            time_stamp=self.get_time(),
            location=GU.Point(0, -3),
            pitch=16.0,
            loudness=13.0,
            duration=150.0,
            level_left=1.0,
            level_right=1.0,
            content=self.probe.color,
            speaker_gender=Symbol("Any"),
            speaker_id=Symbol("Headphones"),
            utterance_id=101,
        )

        self.make_auditory_speech_event(probe_word)

        if self.option["show_debug_messages"]:
            Device_out("present_probe_word_1*\n")

    def present_probe_word2(self):
        if self.option["show_debug_messages"]:
            Device_out("*present_probe_word_2|")

        probe_word = Speech_word(
            name=Symbol("ShapeWord"),
            stream_name=Symbol("RightSpeaker"),
            time_stamp=self.get_time(),
            location=GU.Point(0, 3),
            pitch=16.0,
            loudness=13.0,
            duration=150.0,
            level_left=1.0,
            level_right=1.0,
            content=self.probe.shape,
            speaker_gender=Symbol("Any"),
            speaker_id=Symbol("Headphones"),
            utterance_id=102,
        )

        self.make_auditory_speech_event(probe_word)

        if self.option["show_debug_messages"]:
            Device_out("present_probe_word_2*\n")

    def present_search_objects(self):
        for search_object in self.search_objects:
            self.present_search_object(
                search_object=search_object, name_prefix="Search"
            )

        self.stimulus_onset_time = self.get_time()

        # Which of the presented objects are the targets - name assigned when presented
        self.target = self.search_objects[self.target_num]

    def present_search_object(self, search_object: SearchObject, name_prefix: str):
        # put the symbolic name in from the id number

        obj_name = concatenate_to_Symbol(name_prefix, search_object._id)
        search_object.name = obj_name
        self.make_visual_object_appear(
            obj_name, search_object.location, self.search_object_size
        )
        self.set_visual_object_property(obj_name, Color_c, search_object.color)
        self.set_visual_object_property(obj_name, Shape_c, search_object.shape)

    def remove_stimulus(self):
        if self.option["show_debug_messages"]:
            Device_out("*remove_stimulus|")

        for search_object in self.search_objects:
            self.make_visual_object_disappear(search_object.name)

        if self.option["show_debug_messages"]:
            Device_out("remove_stimulus*\n")

    def setup_next_trial(self):
        if self.trial < self.n_trials:
            if self.option["show_debug_messages"]:
                Device_out("*setup_next_trial|")
            # schedule a new stimulus
            self.state = state.START_TRIAL
            self.schedule_delay_event(1000, Symbol("StartTrial"), Symbol())
            if self.option["show_debug_messages"]:
                Device_out("setup_next_trial*\n")
        else:
            if self.option["show_debug_messages"]:
                Device_out("*shutdown_experiment|")
            # things to do prior to shutdown
            self.finalize_data_output()  # from EpicPyDevice, recommended when task ends!
            self.show_output_stats()
            self.refresh_experiment()  # tidy up for subsequent runs

            # shutting down!
            self.state = self.SHUTDOWN
            self.schedule_delay_event(500)

            if self.option["show_debug_messages"]:
                Device_out("shutdown_experiment*\n")

    def refresh_experiment(self):
        self.trial = 0

        # stimulus generation
        self.search_objects: List[SearchObject] = list()
        self.locations: List[GU.Point] = list()
        self.setup_locations()

        # count the number of time we have a target of each characteristic
        # color, shape, distance
        self.n_target_type = list()

        self.state = state.START

    def handle_Keystroke_event(self, key_name: Symbol):
        if self.state != state.WAITING_FOR_RESPONSE:
            raise Device_exception(
                "Keystroke received while not waiting for a response"
            )

        if self.option["show_debug_messages"]:
            Device_out("\n*handle_keystroke_event|\n")

        eml = self.eyemovement_start_time - self.stimulus_onset_time
        rt = self.get_time() - self.stimulus_onset_time

        # classify the object we looked at
        fixated_obj = self.find_object(self.eyemovement_target_name)
        if (
            fixated_obj.color == self.probe.color
            and fixated_obj.shape == self.probe.shape
        ):
            match_type = MatchType.BOTH
        elif (
            fixated_obj.color == self.probe.color
            and fixated_obj.shape != self.probe.shape
        ):
            match_type = MatchType.COLOR
        elif (
            fixated_obj.color != self.probe.color
            and fixated_obj.shape == self.probe.shape
        ):
            match_type = MatchType.SHAPE
        else:
            match_type = MatchType.NEITHER

        target_distance = GU.cartesian_distance(
            self.target.location, self.probe.location
        )
        distance_type = Distance.NEAR if target_distance < 8.0 else Distance.FAR

        # score the response - should be '/' if correct
        if key_name == Symbol("/"):
            assert match_type == MatchType.BOTH

            if self.option["show_trial_data"]:
                # data display for normal output window
                s = (
                    f"\n{self.processor_info()} Correct trial: "
                    f"{self.trial} {'near' if self.distance_index else 'far'} "
                    f"target EML: {eml} RT: {rt}\n"
                )
                Device_out(s)
        else:
            # self.n_incorrect[self.distance_index] += 1
            if self.option["show_trial_data"]:
                # data display for normal output window
                s = f"\n{self.processor_info()} Incorrect trial: {self.trial}\n"
                Device_out(s)

        data = (
            self.task_name,
            self.trial,
            eml,
            distance_type.name,
            "/",
            str(key_name),
            "Keyboard",
            rt,
            "Correct" if key_name == Symbol("/") else "Incorrect",
            match_type.name,
            self.device_name,
            self.rule_filename,
            datetime.now().ctime(),
        )
        self.data_writer.writerow(data)

        if self.option["show_debug_messages"]:
            Device_out("\n|handle_keystroke_event*\n")

        # make the stimulus disappear
        self.remove_stimulus()
        self.trial += 1

        self.state = state.FINISH_TRIAL
        self.schedule_delay_event(100, Symbol("FinishTrial"), Symbol())

    def handle_Eyemovement_Start_event(
        self, target_name: Symbol, new_location: GU.Point
    ):
        self.eyemovement_start_time = self.get_time()
        self.eyemovement_target_name = Symbol(target_name)
        pass

    def handle_Eyemovement_End_event(self, target_name: Symbol, new_location: GU.Point):
        pass

    def find_object(self, name: Symbol) -> SearchObject:
        for search_object in self.search_objects:
            if search_object.name == name:
                return search_object

        # here only if not found
        raise Device_exception(f"Looked at an unknown search object ('{name}')")


    def show_output_stats(self):
        Device_out("\n*** End of experiment! ***\n")

        self.stats_write("Findlay 1997 Results", color="Orange")
        self.stats_write(f"<h4>Total Trials: {self.trial}</h4>", color="Green")

        if not self.option.get("show_task_statistics", False):
            return

        # Target distribution table (keeps your original ordering)
        self.stats_write("<h4>Target distribution:</h4>", color="Green")
        rows = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    rows.append([i, j, k, self.n_target_type[i][j][k]])
        tgt_table = pd.DataFrame(rows, columns=["Color", "Shape", "Dist", "N"])
        self.stats_write(tgt_table)

        # ----- Load data with local precision setting
        with pd.option_context("display.precision", 3):
            data = pd.read_csv(self.data_filepath)

        # ----- Recode accuracy (vectorized; tolerant to case/whitespace)
        if "ACC" in data.columns:
            acc_is_correct = data["ACC"].astype(str).str.strip().str.lower().eq("correct")
            data["ACC"] = acc_is_correct.astype(float) * 100.0
        else:
            self.stats_write("Missing 'ACC' column; skipping recode.", color="Red")

        # ----- Tables
        self.stats_write("<h4>Overall:</h4>", color="Green")
        try:
            cols = [c for c in ("RT", "EML", "ACC") if c in data.columns]
            if not cols:
                raise KeyError("None of RT/EML/ACC present")
            tbl = data[cols].agg(["mean", "count"])
            # match your compact integer display
            tbl = tbl.round(0).astype(int)
            self.stats_write(tbl.transpose())
        except Exception as e:
            self.stats_write(f"Error creating table: {e}", color="Red")

        self.stats_write("<h4>Near vs Far:</h4>", color="Green")
        try:
            need = {"DistType"} | set(cols)
            if not need.issubset(data.columns):
                raise KeyError(f"Missing columns: {sorted(need - set(data.columns))}")
            tbl = (
                data.groupby(["DistType"], as_index=True)[list(cols)]
                    .agg(["mean", "count"])
                    .round(0).astype(int)
            )
            self.stats_write(tbl.transpose())
        except Exception as e:
            self.stats_write(f"Error creating table: {e}", color="Red")

        self.stats_write("<h4>MatchType:</h4>", color="Green")
        try:
            need = {"MatchType"} | set(cols)
            if not need.issubset(data.columns):
                raise KeyError(f"Missing columns: {sorted(need - set(data.columns))}")
            tbl = (
                data.groupby(["MatchType"], as_index=True)[list(cols)]
                    .agg(["mean", "count"])
                    .round(0).astype(int)
            )
            self.stats_write(tbl.transpose())
        except Exception as e:
            self.stats_write(f"Error creating table: {e}", color="Red")

        self.stats_write("<h4>DistType vs MatchType:</h4>", color="Green")
        try:
            need = {"DistType", "MatchType"} | set(cols)
            if not need.issubset(data.columns):
                raise KeyError(f"Missing columns: {sorted(need - set(data.columns))}")
            tbl = (
                data.groupby(["DistType", "MatchType"], as_index=True)[list(cols)]
                    .agg(["mean", "count"])
                    .round(0).astype(int)
            )
            self.stats_write(tbl.transpose())
        except Exception as e:
            self.stats_write(f"Error creating table: {e}", color="Red")

        # ----- Bar plots
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.5)  # paper, notebook, talk, poster

        def _barplot(y_col: str, acc_value: float, title: str, ylabel: str):
            try:
                need = {"MatchType", "DistType", y_col, "ACC"}
                if not need.issubset(data.columns):
                    missing = sorted(need - set(data.columns))
                    raise KeyError(f"Missing columns for plot: {missing}")

                mask = data["ACC"] == acc_value
                df = data.loc[mask, ["MatchType", "DistType", y_col]].copy()
                if df.empty:
                    raise ValueError(f"No rows where ACC == {acc_value} to plot.")

                fig, ax = plt.subplots(figsize=(7, 4), dpi=96)
                sns.barplot(
                    x="MatchType",
                    y=y_col,
                    hue="DistType",
                    data=df,
                    capsize=0.1,
                    ax=ax,
                )
                ax.set_title(title)
                ax.set_xlabel("Match Type")
                ax.set_ylabel(ylabel)
                plt.tight_layout()
                self.stats_write(fig)
                plt.close(fig)
            except Exception as e:
                self.stats_write(f"Error creating barplot ({y_col}, ACC={acc_value}): {e}", color="Red")

        self.stats_write("<h4>BarPlots:</h4>", color="Green")
        _barplot("RT",  100.0, "Correct Mean RT by MatchType and DistanceType",    "Mean Response Time (ms)")
        _barplot("RT",    0.0, "Incorrect Mean RT by MatchType and DistanceType",  "Mean Response Time (ms)")
        _barplot("EML", 100.0, "Correct Mean EML by MatchType and DistanceType",   "Mean Response Time (ms)")
        _barplot("EML",   0.0, "Incorrect Mean EML by MatchType and DistanceType", "Mean Response Time (ms)")

        # final cleanup
        plt.close("all")
