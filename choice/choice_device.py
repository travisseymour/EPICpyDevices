import textwrap
from typing import Any

from pathlib import Path
from enum import Enum, auto
import random
from datetime import datetime

from epicpydevice import epicpy_device_base
from epicpydevice.output_tee import Output_tee
from epicpydevice.output_tee_globals import (Device_out, Exception_out, Debug_out)
from epicpydevice.epic_statistics import Mean_accumulator
from epicpydevice.symbol_utilities import concatenate_to_Symbol, get_nth_Symbol
from epicpydevice.device_exception import Device_exception
from epicpydevice.speech_word import Speech_word
import epicpydevice.geometric_utilities as GU
from epicpydevice.symbol import Symbol
from epicpydevice.standard_symbols import (
    Red_c,
    Green_c,
    Blue_c,
    Yellow_c,
    Black_c,
    Text_c,
    Color_c,
    Shape_c,
    Circle_c,
)

import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple


class state(Enum):
    START = auto()
    START_TRIAL = auto()
    PRESENT_FIXATION = auto()
    REMOVE_FIXATION = auto()
    PRESENT_STIMULUS = auto()
    WAITING_FOR_RESPONSE = auto()
    DISCARD_STIMULUS = auto()
    FINISH_TRIAL = auto()
    SHUTDOWN = auto()


USING_OLDER_EPICLIB = False

# EpicPy will expect all devices to be of class "EpicDevice" and subclassed from
# "EpicPyDevice" or "epicpy_device.EpicPyDevice"


class EpicDevice(epicpy_device_base.EpicPyDevice):
    def __init__(self, ot: Output_tee, parent: Any, device_folder: Path):
        epicpy_device_base.EpicPyDevice.__init__(
            self,
            parent=parent,
            ot=ot,
            device_name="Choice_Device_v2025.3.19",
            device_folder=device_folder,
        )
        # NOTE: ot is not being used, just use Device_out(...)
        # self.device_name = "Choice_Device_v2025.3.19"
        self.condition_string = "10 4 Hard Draft"  # from EpicPyDevice, default is ''
        self.n_trials = 10
        self.color_count = 4
        self.task_difficulty = "Hard"
        self.tag_str = "Draft"

        # differentiator in case this data is combined with Detection data
        self.task_name = "Choice"

        self.trial = 0
        self.run_id = self.unique_id()

        self.current_vrt = Mean_accumulator()

        # EPIC Simulation controller will know device is finished when
        #   self.state == self.SHUTDOWN
        self.state = state.START  # from EpicPyDevice, default is 0
        self.SHUTDOWN = state.SHUTDOWN  # from EPICPyDevice, default is 1000

        self.vresponse_made = False
        self.vstims = [Red_c, Green_c, Blue_c, Yellow_c]
        self.vresps = [Symbol("U"), Symbol("I"), Symbol("O"), Symbol("P")]
        self.vstimresp = list(zip(self.vstims, self.vresps))
        self.vstim_color = None
        self.correct_vresp = None
        self.vstim_name = None
        self.vstim_xloc = None
        self.vstim_onset = 0

        self.reparse_conditionstring = False  # from EpicPyDevice, default is False

        # device constants
        self.Warning_c = Symbol("#")
        self.VWarn_c = Symbol("Fixation")
        self.VStim_c = Symbol("Stimulus")
        self.ColorWord_c = Symbol("ColorWord")
        self.Center_screen_c = Symbol("Center_screen")

        # experiment constants
        self.wstim_location_c = GU.Point(0.0, 0.0)
        self.wstim_size_c = GU.Size(1.0, 1.0)
        self.vstim_size_c = GU.Size(2.0, 2.0)
        self.intertrialinterval_c = 5000

        # Optionally expose some boolean device options to the user via the GUI.
        self.option = dict()  # from EpicPyDevice, default is dict()
        # useful for showing debug information during development
        self.option["show_debug_messages"] = True
        # useful for showing device state info during trial run
        self.option["show_trial_events"] = True
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
            "RunID",
            "Trial",
            "Task",
            "Difficulty",
            "StimColor",
            "StimPos",
            "CorrectResponse",
            "Response",
            "Modality",
            "RT",
            "Accuracy",
            "Tag",
            "Device",
            "Rules",
            "Date",
        )

        # assumes device file is next to folder called images!
        self.show_view_background(
            view_type="visual", file_name="donders_monitor.png", scaled=True
        )
        self.show_view_background(
            view_type="auditory", file_name="donders_tones.png", scaled=True
        )

        if self.option["show_task_description"]:
            self.describe_task()

    def parse_condition_string(self):
        error_msg = (
            f"{self.condition_string}\n Should be: space-delimited "
            f"trials(int > 0) colors(1-4) difficulty(Easy or Hard) "
            f"tag(any string)"
        )

        # returns items from space delimited condition string as strings.
        # Uses self.get_param_list() instead of self.condition_string.split(' ') to
        # ensure any accidentally remaining range information is ignored.
        params = self.get_param_list()

        try:
            assert len(params) == 4, "Incorrect condition string: "
            trials, colors, difficulty, tag = params
            difficulty = difficulty.lower()
            trials = int(trials) if str(trials).isdigit() else 0
            colors = int(colors) if str(colors).isdigit() else 0
            assert trials > 0, "Number of trials must be positive: "
            assert 1 <= colors <= 4, "Colors must be between 1 and 4"
            assert difficulty in (
                "easy",
                "hard",
            ), "Task difficulty must be 'Easy' or 'Hard'"
        except AssertionError as e:
            raise Device_exception(
                f"Error Parsing the Condition String: {e} | {error_msg}"
            )

        self.color_count = colors
        self.n_trials = trials
        self.task_difficulty = difficulty
        self.tag_str = tag

        self.reparse_conditionstring = False

    def initialize(self):
        """
        Initializes run. Called whenever model is stopped and started anew.
        I.e., NOT called, when model is paused and resumed.
        """
        self.vresponse_made = False
        self.trial = 0
        self.run_id = self.unique_id()
        self.state = state.START
        self.current_vrt.reset()

        # from EpicPyDevice, recommended for your initialize method!
        self.init_data_output()

    def describe_task(self):
        description = f"""
        ******************************************************************
        f"Initializing Device: {self.device_name.replace('_', ' ')} 
        f"Current Parameter String: {self.condition_string}
        ******************************************************************
        
        Events
        ------
         - Trial Start
         - Warning Signal (Simultaneous Visual and Auditory)
           Visual: "#" (1000 ms)
           Auditory: 'Beep' for 300 ms & 
         - Blank Screen (500-1300 ms random)
         - Stimulus (Simultaneous Visual and Auditory) for 500 ms
           Visual: Red, Green, Blue, or Yellow filled circle 
           Auditory: "Ding", "Dong", "Bing", or "Bong"
         - Waits for keyboard button press or vocal utterance
           Correct Responses
           Keyboard: U,I,O,P for R,G,B,Y, respectively
           Vocal: "U","I","O","P" for R,G,B,Y, respectively
         - Inter-Trial Interval, Blank Screen, for 5000 ms
        
        Conditions:
        ----------
         - Easy: X position is at screen center (x = 0)
         - Hard: X position randomly selected from [-2, -1, 0, 1, 2]
        
        Parameter String:
        ----------------
         - Structure: TRIALS NUM_COLORS DIFFICULTY TAG
         - Values: Trials: Integer > 0 
                   Num Colors: Integer 1-4
                   Difficulty: Easy or Hard
                   Tag: Any Word (simply added to TAG column in data file)
         - Default: 10 4 Hard Draft
        ******************************************************************
        """
        Device_out(textwrap.dedent(description))

    def handle_Start_event(self):
        """
        You have to get the ball rolling with a first time-delayed event,
        nothing happens until you do.
        """
        self.schedule_delay_event(500)

    def handle_Stop_event(self):
        """
        called after the stop_simulation function (which is part of the base device class)
        """
        Device_out(f"{self.processor_info()} received Stop_event\n")
        self.finalize_data_output()  # from EpicPyDevice, recommended
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
            if self.option["show_trial_events"]:
                Device_out("********-->STATE: START\n")
            self.state = state.START_TRIAL
            self.schedule_delay_event(500, Symbol("StartTrial"), Symbol())
        elif self.state == state.START_TRIAL and _type == Symbol("StartTrial"):
            if self.option["show_trial_events"]:
                Device_out("********-->STATE: START_TRIAL\n")
            self.vresponse_made = False
            self.start_trial()
            self.state = state.PRESENT_FIXATION
            self.schedule_delay_event(100, Symbol("PresentFixation"), Symbol())
        elif self.state == state.PRESENT_FIXATION and _type == Symbol(
            "PresentFixation"
        ):
            if self.option["show_trial_events"]:
                Device_out("********-->STATE: PRESENT_FIXATION\n")
            self.present_fixation_point()
            self.state = state.REMOVE_FIXATION
            self.schedule_delay_event(1000, Symbol("RemoveFixation"), Symbol())
        elif self.state == state.REMOVE_FIXATION and _type == Symbol("RemoveFixation"):
            if self.option["show_trial_events"]:
                Device_out("********-->STATE: REMOVE_FIXATION\n")
            self.remove_fixation_point()
            self.state = state.PRESENT_STIMULUS
            stimwaittime = random.randint(800, 1300)
            self.schedule_delay_event(stimwaittime, Symbol("PresentStimulus"), Symbol())
        elif self.state == state.PRESENT_STIMULUS:
            if self.option["show_trial_events"]:
                Device_out("********-->STATE: PRESENT_STIMULUS\n")
            self.present_stimulus()
            self.state = state.WAITING_FOR_RESPONSE
            self.schedule_delay_event(100, Symbol("WaitForResp"), Symbol())
        elif self.state == state.WAITING_FOR_RESPONSE and _type == Symbol(
            "WaitForResp"
        ):
            if self.option["show_trial_events"]:
                Device_out("********-->STATE: WAITING_FOR_RESPONSE\n")
            # nothing to do, just wait
            # note: next state and schedule_delay_event sent by response handler!
        elif self.state == state.DISCARD_STIMULUS:
            if self.option["show_trial_events"]:
                Device_out("********-->STATE: DISCARD_STIMULUS\n")
            self.remove_stimulus()
            self.state = state.FINISH_TRIAL
            self.schedule_delay_event(
                self.intertrialinterval_c, Symbol("FinishTrial"), Symbol()
            )
        elif self.state == state.FINISH_TRIAL and _type == Symbol("FinishTrial"):
            if self.option["show_trial_events"]:
                Device_out("********-->STATE: FINISH_TRIAL\n")
            self.setup_next_trial()
        elif self.state == state.SHUTDOWN:
            if self.option["show_trial_events"]:
                Device_out("********-->STATE: SHUTDOWN\n")
            self.stop_simulation()  # doesn't do anything in EpicPy...could just omit
        else:
            raise Device_exception(
                f"Device delay event in unknown or improper device state: {self.state}"
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

        self.wstim_v_name = concatenate_to_Symbol(self.VWarn_c, self.trial)
        self.make_visual_object_appear(
            self.wstim_v_name, self.wstim_location_c, self.wstim_size_c
        )
        self.set_visual_object_property(self.wstim_v_name, Text_c, self.Warning_c)
        self.set_visual_object_property(self.wstim_v_name, Color_c, Black_c)

        # no need to keep ref, sound will remove itself after a delay
        wstim_snd_name = concatenate_to_Symbol("WarningSound", self.trial)
        self.make_auditory_sound_event(
            wstim_snd_name, Symbol("Signal"), GU.Point(0, -5), Symbol("Beep"), 12, 300
        )

        wstim_snd_name = Symbol(f"WarningSpeech{self.trial}")

        warning_word = Speech_word(
            name=wstim_snd_name,
            stream_name=Symbol("ComputerSpeaker"),
            time_stamp=self.get_time(),
            location=GU.Point(0, -10),
            pitch=16.0,
            loudness=13.0,
            duration=500.0,
            level_left=1.0,
            level_right=1.0,
            content=Symbol("Warning"),
            speaker_gender=Symbol("Any"),
            speaker_id=Symbol("Computer"),
            utterance_id=100,
        )

        self.make_auditory_speech_event(warning_word)

        if self.option["show_debug_messages"]:
            Device_out("present_fixation*\n")

    def remove_fixation_point(self):
        if self.option["show_debug_messages"]:
            Device_out("*remove_fixation|")

        self.make_visual_object_disappear(self.wstim_v_name)

        if self.option["show_debug_messages"]:
            Device_out("remove_fixation*\n")

    def present_stimulus(self):
        if self.option["show_debug_messages"]:
            Device_out("*present_stimulus|")

        self.vstim_color, self.correct_vresp = random.choice(self.vstimresp)
        self.vstim_name = concatenate_to_Symbol(self.VStim_c, self.trial)

        if self.task_difficulty == "easy":
            self.vstim_xloc = 0.0
        else:
            self.vstim_xloc = random.choice((-2.0, -1.0, 0.0, 1.0, 2.0))

        self.make_visual_object_appear(
            self.vstim_name, GU.Point(self.vstim_xloc, 0.0), self.vstim_size_c
        )
        self.set_visual_object_property(self.vstim_name, Shape_c, Circle_c)
        self.set_visual_object_property(self.vstim_name, Color_c, self.vstim_color)

        self.vstim_onset = self.get_time()
        self.vresponse_made = False

        # no need to keep ref, sound will remove itself after a delay
        vstim_snd_name = concatenate_to_Symbol("StimulusSound", self.trial)
        stim_sound = {"Red": "Ding", "Green": "Dong", "Blue": "Bing", "Yellow": "Bong"}

        self.make_auditory_sound_event(
            vstim_snd_name,
            Symbol("Signal"),
            GU.Point(0, 10),
            Symbol(stim_sound[str(self.vstim_color)]),
            12,
            500,
            500,
        )

        if self.option["show_debug_messages"]:
            Device_out("present_stimulus*\n")

    def remove_stimulus(self):
        if self.option["show_debug_messages"]:
            Device_out("*remove_stimulus|")

        self.make_visual_object_disappear(self.vstim_name)

        if self.option["show_debug_messages"]:
            Device_out("remove_stimulus*\n")

    def setup_next_trial(self):
        if self.trial < self.n_trials:
            if self.option["show_debug_messages"]:
                Device_out("*setup_next_trial|")
            self.state = state.START_TRIAL
            self.schedule_delay_event(300, Symbol("StartTrial"), Symbol())
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
        self.vresponse_made = False
        self.trial = 0
        self.state = state.START
        self.current_vrt.reset()

    def output_statistics(self):
        s = (
            f"Task Name: {self.task_name}  Task Difficulty: {self.task_difficulty}"
            f"  Trials Run: {self.trials}/{self.n_trials}\n"
        )
        Device_out(s)

        s = f"N = {self.current_vrt.get_n()}, Mean RT = {self.current_vrt.get_mean()}"
        Device_out(f"{s}\n(note: Mean excludes 1st trial)\n\n")

    def handle_Keystroke_event(self, key_name: Symbol):
        if self.vresponse_made:
            return

        if self.option["show_debug_messages"]:
            Device_out("\n*handle_keystroke_event...\n")

        rt = self.get_time() - self.vstim_onset
        if key_name == self.correct_vresp:
            outcome = "CORRECT"
            if self.trial > 1:
                self.current_vrt.update(rt)
        else:
            outcome = "INCORRECT"

        if self.option["show_trial_data"]:
            # data display for normal output window
            s = (
                f"\nTrial: {self.trial} | Task: {self.task_name} | Difficulty: "
                f"{self.task_difficulty} | RT: {rt} | StimColor: {self.vstim_color} | "
                f"StimPos: {self.vstim_xloc} | Modality: Keyboard | "
                f"Response: {key_name} | CorrectResponse: {self.correct_vresp} | "
                f"Accuracy: {outcome}\n"
            )

            Device_out(s)

        # real data saved to file
        if self.data_file:
            # note: order of values dictated by names in self.data_header
            data = (
                self.run_id,
                self.trial,
                self.task_name,
                self.task_difficulty,
                self.vstim_color,
                self.vstim_xloc,
                self.correct_vresp,
                key_name,
                "Keyboard",
                rt,
                outcome,
                self.tag_str,
                self.device_name,
                self.rule_filename,
                datetime.now().ctime(),
            )
            self.data_writer.writerow(data)

        self.vresponse_made = True

        if self.option["show_debug_messages"]:
            Device_out("\nhandle_keystroke_event*\n")

        self.state = state.DISCARD_STIMULUS
        self.schedule_delay_event(500)

    def handle_Vocal_event(self, vocal_input: Symbol, duration: int = 0):
        if self.vresponse_made:
            return

        if self.option["show_debug_messages"]:
            Device_out("\n*handle_vocal_event...\n")

        rt = self.get_time() - self.vstim_onset
        if vocal_input == self.correct_vresp:
            outcome = "CORRECT"
            if self.trial > 1:
                self.current_vrt.update(rt)
        else:
            outcome = "INCORRECT"

        if self.option["show_trial_data"]:
            # data display for normal output window
            s = (
                f"\nTrial: {self.trial} | Task: {self.task_name} | "
                f"Difficulty: {self.task_difficulty} | "
                f"RT: {rt} | StimColor: {self.vstim_color} | "
                f"StimPos: {self.vstim_xloc} | Modality: Vocal | "
                f"Response: {vocal_input} | CorrectResponse: {self.correct_vresp} | "
                f"Accuracy: {outcome}\n"
            )

            Device_out(s)

        # real data saved to file
        if self.data_file:
            # note: order of values dictated by names in self.data_header
            data = (
                self.run_id,
                self.trial,
                self.task_name,
                self.task_difficulty,
                self.vstim_color,
                self.vstim_xloc,
                self.correct_vresp,
                vocal_input,
                "Voice",
                rt,
                outcome,
                self.tag_str,
                self.device_name,
                Path(self.rule_filename).name,
                datetime.now().ctime(),
            )
            self.data_writer.writerow(data)

        self.vresponse_made = True

        if self.option["show_debug_messages"]:
            Device_out("\nhandle_vocal_event*\n")

        self.state = state.DISCARD_STIMULUS
        self.schedule_delay_event(500)

    def handle_Eyemovement_Start_event(
        self, target_name: Symbol, new_location: GU.Point
    ):
        ...

    def handle_Eyemovement_End_event(self, target_name: Symbol, new_location: GU.Point):
        ...

    def show_output_stats(self):
        # use pandas to load in data run so far (include all data from previous runs)
        # (assumes self.finalize_data_output() has been called)
        # Use pandas option locally (won't persist globally across your whole process)
        with pd.option_context("display.precision", 3):
            data = pd.read_csv(self.data_filepath)

        # Summary line (same format as before)
        correct_mask = data["Accuracy"].eq("CORRECT")
        correct_data = data.loc[correct_mask]
        correct_trials_N = correct_data["Accuracy"].shape[0]
        all_trials_N = data["Accuracy"].shape[0]

        mean_rt = correct_data["RT"].mean() if correct_trials_N else None
        accuracy = (correct_trials_N / all_trials_N * 100) if all_trials_N else 0.0

        self.stats_write(
            f"N={all_trials_N}, "
            f"CORRECT={correct_trials_N} "
            f"INCORRECT={all_trials_N - correct_trials_N} "
            f'MEAN_RT={"NA" if mean_rt is None else int(mean_rt)} '
            f"ACCURACY={accuracy:0.2f}%"
        )

        if not self.option.get("show_task_statistics", False):
            return

        # Route to 1-way vs 2-way logic based on Difficulty values (easy/hard only)
        if "Difficulty" in data.columns:
            diffs = (
                data["Difficulty"]
                .dropna()
                .astype(str)
                .str.lower()
                .unique()
                .tolist()
            )
            if set(diffs) == {"easy", "hard"}:
                self.show_output_stats_2way(data)
                return

        self.show_output_stats_1way(data)

    def show_output_stats_1way(self, data: pd.DataFrame):
        try:
            # Filter to correct trials, work on a copy to avoid chained-assign warnings
            data = data.loc[data["Accuracy"].eq("CORRECT")].copy()

            # Colors from device config (unchanged behavior)
            colors = [str(vstim)[0] for vstim in self.vstims]
            colors = colors[: self.color_count]

            # Condition label if present, else "All"
            cond = (
                str(data["Difficulty"].iloc[0]).title()
                if "Difficulty" in data.columns and not data["Difficulty"].empty
                else "All"
            )

            self.stats_write(
                f"<h4>Choice Task<br>ANOVA: RT by StimColor ({cond} Condition)</h4>",
                color="Orange",
            )

            # Means table
            try:
                table = (
                    data.groupby(["StimColor"], as_index=True)["RT"]
                    .agg(["mean", "count"])
                    .astype(int)
                )
                self.stats_write(table.transpose())
            except Exception as e:
                self.stats_write(
                    f"Not enough data to create means table: {e}", color="Red"
                )

            # Repeated-measures ANOVAs
            try:
                # At least two colors, enough runs
                err_msg = "Data must contain StimColor with at least 2 colors."
                assert data["StimColor"].nunique(dropna=True) >= 2, err_msg
                err_msg = (
                    "For an ANOVA, the data must contain at least 3 RunIDs "
                    "from at least 3 separate runs."
                )
                assert data["RunID"].nunique(dropna=True) >= 3, err_msg

                # RT ~ StimColor (within = StimColor, subject = RunID)
                aov = pg.rm_anova(data=data, dv="RT", subject="RunID", within="StimColor")
                _cosmetic = ["p-GG-corr", "eps", "sphericity", "W-spher", "p-spher"]
                drop_cols = [c for c in _cosmetic if c in aov.columns]
                if drop_cols:
                    aov = aov.drop(columns=drop_cols)
                self.stats_write(aov)

                # RT ~ StimPos if present
                if "StimPos" in data.columns and data["StimPos"].nunique(dropna=True) >= 2:
                    aov2 = pg.rm_anova(data=data, dv="RT", subject="RunID", within="StimPos")
                    drop_cols = [c for c in _cosmetic if c in aov2.columns]
                    if drop_cols:
                        aov2 = aov2.drop(columns=drop_cols)
                    self.stats_write(aov2)
            except AssertionError as e:
                self.stats_write(
                    f"Not enough data to run mixed-model anova: {e}", color="Red"
                )
            except Exception as e:
                self.stats_write(f"Unable to run mixed-model anova: {e}", color="Red")

            # ---------- Plots ----------
            # Consistent ordering and palette
            color_order = ["Red", "Green", "Blue", "Yellow"]
            color_map = {"Red": "red", "Green": "green", "Blue": "blue", "Yellow": "yellow"}

            # sns.set_context(style="whitegrid")
            sns.set_context("paper", font_scale=1.5)

            # Primary bar plot: RT by StimColor
            fig, ax = plt.subplots(figsize=(7, 4), dpi=96)
            my_plot = sns.barplot(
                x="StimColor",
                y="RT",
                hue="StimColor",
                data=data,
                order=color_order,
                palette=color_map,
                capsize=0.1,
                ax=ax,
            )

            try:
                highest_mean = int(data.groupby("StimColor")["RT"].mean().max()) + 50
            except ValueError:
                highest_mean = 400
            ymin = 150
            ax.set_ylim(ymin, max(ymin, highest_mean))
            ticks = list(range(ymin, max(ymin, highest_mean) + 1, 50))
            ax.set_yticks(ticks)

            if ax.legend_:
                ax.legend_.remove()

            ax.set_title(f"Mean RT by Stimulus Color, {cond} Condition")
            ax.set_xlabel("Stimulus Color")
            ax.set_ylabel("Mean Response Time (ms)")
            plt.tight_layout()
            self.stats_write(my_plot.get_figure())

            # OPTIONAL GRAPH BY RULES (only if >1 level)
            if "Rules" in data.columns and data["Rules"].nunique(dropna=True) > 1:
                fig, ax = plt.subplots(figsize=(7, 4), dpi=96)
                my_plot = sns.barplot(
                    x="StimColor",
                    y="RT",
                    hue="Rules",
                    data=data,
                    order=color_order,
                    capsize=0.1,
                    ax=ax,
                )
                try:
                    highest_mean = int(
                        data.groupby(["StimColor"])["RT"].mean().max()
                    ) + 50
                except ValueError:
                    highest_mean = 200
                ax.set_ylim(ymin, max(ymin, highest_mean))
                ticks = list(range(ymin, max(ymin, highest_mean) + 1, 50))
                ax.set_yticks(ticks)

                ax.set_title(f"Mean RT by Stimulus Color & Model, {cond} Condition")
                ax.set_xlabel("Stimulus Color")
                ax.set_ylabel("Mean Response Time (ms)")
                plt.tight_layout()
                self.stats_write(my_plot.get_figure())

            # OPTIONAL student_data comparison
            student_data_path = Path(self.data_filepath).parent / "student_data.csv"
            if student_data_path.is_file():
                try:
                    student_data = pd.read_csv(student_data_path)
                    combined = pd.concat([data, student_data], ignore_index=True)

                    fig, ax = plt.subplots(figsize=(7, 4), dpi=96)
                    my_plot = sns.barplot(
                        x="StimColor",
                        y="RT",
                        hue="Rules",
                        data=combined,
                        order=color_order,
                        capsize=0.1,
                        ax=ax,
                    )
                    try:
                        highest_mean = int(
                            combined.groupby(["StimColor", "Rules"])["RT"].mean().max()
                        ) + 50
                    except ValueError:
                        highest_mean = 200
                    ax.set_ylim(ymin, max(ymin, highest_mean))
                    ax.set_yticks(list(range(ymin, max(ymin, highest_mean) + 1, 50)))

                    ax.set_title("Mean RT by Stimulus Color Hard (Student Data)")
                    ax.set_xlabel("Stimulus Color")
                    ax.set_ylabel("Mean Response Time (ms)")
                    plt.tight_layout()
                    self.stats_write(my_plot.get_figure())
                except Exception as e:
                    self.stats_write(
                        f'<br><font color="Red">Error while trying to create student comparison graph: {e}</font><br>'
                    )

            # OPTIONAL class_data comparison
            class_data_path = Path(self.data_filepath).parent / "class_data.csv"
            if class_data_path.is_file():
                try:
                    class_data = pd.read_csv(class_data_path)
                    combined = pd.concat([data, class_data], ignore_index=True)

                    fig, ax = plt.subplots(figsize=(7, 4), dpi=96)
                    my_plot = sns.barplot(
                        x="StimColor",
                        y="RT",
                        hue="Rules",
                        data=combined,
                        order=color_order,
                        capsize=0.1,
                        ax=ax,
                    )
                    try:
                        highest_mean = int(
                            combined.groupby(["StimColor", "Rules"])["RT"].mean().max()
                        ) + 50
                    except ValueError:
                        highest_mean = 200
                    ax.set_ylim(ymin, max(ymin, highest_mean))
                    ax.set_yticks(list(range(ymin, max(ymin, highest_mean) + 1, 50)))

                    ax.set_title("Mean RT by Stimulus Color Hard (Class Data)")
                    ax.set_xlabel("Stimulus Color")
                    ax.set_ylabel("Mean Response Time (ms)")
                    plt.tight_layout()
                    self.stats_write(my_plot.get_figure())
                except Exception as e:
                    self.stats_write(
                        f'<br><font color="Red">Error while trying to create class comparison graph: {e}</font><br>'
                    )

            # Eccentricity plot if available
            if "StimPos" in data.columns:
                fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=96)
                my_eccentricity_plot = sns.barplot(
                    x="StimPos",
                    y="RT",
                    hue="StimPos",
                    data=data,
                    capsize=0.1,
                    ax=ax2,
                    legend=False,
                )
                ax2.set_title(f"Mean RT by Eccentricity, {cond} Condition")
                ax2.set_xlabel("Eccentricity")
                ax2.set_ylabel("Mean Response Time (ms)")
                plt.tight_layout()
                self.stats_write(my_eccentricity_plot.get_figure())

            plt.close("all")

        except Exception as e:
            self.stats_write(f"Error showing output stats: {e}", color="Red")

    def show_output_stats_2way(self, data: pd.DataFrame):
        try:
            # Filter to correct trials, copy
            data = data.loc[data["Accuracy"].eq("CORRECT")].copy()

            # Colors from device config (unchanged behavior)
            colors = [str(vstim)[0] for vstim in self.vstims]
            colors = colors[: self.color_count]

            self.stats_write(
                f"<h4>Choice Task<br>ANOVA Difficulty [Hard, Easy] X StimColor "
                f"{colors}</h4>",
                color="Orange",
            )

            # Means table
            try:
                table = (
                    data.groupby(["Difficulty", "StimColor"], as_index=True)["RT"]
                    .agg(["mean", "count"])
                    .astype(int)
                )
                self.stats_write(table.transpose())
            except Exception as e:
                self.stats_write(
                    f"Not enough data to create means table: {e}", color="Red"
                )

            # Mixed ANOVA (between: Difficulty; within: StimColor)
            try:
                err = "Data must contain both 'Hard' and 'Easy' Difficulty."
                assert set(
                    data["Difficulty"].dropna().astype(str).str.lower().unique().tolist()
                ) == {"hard", "easy"}, err

                err = "Data must contain StimColor with at least 2 colors."
                assert data["StimColor"].nunique(dropna=True) >= 2, err

                err = (
                    "Data must contain at least 4 RunIDs from at least 2 "
                    "separate runs of 'Easy' Difficulty, and at least 2 separate "
                    "runs of 'Hard' Difficulty."
                )
                assert data["RunID"].nunique(dropna=True) >= 4, err

                aov = pg.mixed_anova(
                    data=data,
                    dv="RT",
                    subject="RunID",
                    between="Difficulty",
                    within="StimColor",
                )
                _cosmetic = ["p-GG-corr", "eps", "sphericity", "W-spher", "p-spher"]
                drop_cols = [c for c in _cosmetic if c in aov.columns]
                if drop_cols:
                    aov = aov.drop(columns=drop_cols)
                self.stats_write(aov)
            except AssertionError as e:
                self.stats_write(
                    f"Not enough data to run mixed-model anova: {e}", color="Red"
                )
            except Exception as e:
                self.stats_write(f"Unable to run mixed-model anova: {e}", color="Red")

            # ---------- Plots ----------
            # sns.set_context(style="whitegrid")
            sns.set_context("notebook", font_scale=1.5)

            fig, ax = plt.subplots(figsize=(7, 5), dpi=96)
            my_plot = sns.barplot(
                x="StimColor",
                y="RT",
                hue="Difficulty",
                hue_order=("hard", "easy"),
                palette={"easy": "grey", "hard": "black"},
                data=data,
                capsize=0.1,
                ax=ax,
            )

            # Rebuild legend from bar containers (as in your original), but guard it
            try:
                handles = [tuple(bar_group) for bar_group in ax.containers]
                labels = [bar_group.get_label() for bar_group in ax.containers]
                if ax.legend_:
                    ax.legend_.remove()
                ax.legend(
                    handles=handles,
                    labels=labels,
                    title="Difficulty",
                    handlelength=4,
                    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
                )
            except Exception:
                # Fall back to default legend if structure differs across seaborn versions
                if not ax.legend_:
                    ax.legend(title="Difficulty")

            ax.set_title("Mean RT by Difficulty and Stimulus Color")
            ax.set_xlabel("Stimulus Color")
            ax.set_ylabel("Mean Response Time (ms)")
            sns.despine()
            plt.tight_layout()
            self.stats_write(my_plot.get_figure())

            # Eccentricity plot if available
            if "StimPos" in data.columns:
                fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=96)
                my_eccentricity_plot = sns.barplot(
                    x="StimPos",
                    y="RT",
                    hue="Difficulty",
                    data=data,
                    capsize=0.1,
                    ax=ax2,
                    legend=False,
                )
                ax2.set_title("Mean RT by Eccentricity by Condition")
                ax2.set_xlabel("Eccentricity")
                ax2.set_ylabel("Mean Response Time (ms)")
                plt.tight_layout()
                self.stats_write(my_eccentricity_plot.get_figure())

            plt.close("all")

        except Exception as e:
            self.stats_write(f"Error showing output stats: {e}", color="Red")

    def __getattr__(self, name):
        def _missing(*args, **kwargs):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("A missing method was called.")
            print(f"The object was {self}, the method was {name}. ")
            print(f"It was called with {args} and {kwargs} as arguments\n")

        return _missing
