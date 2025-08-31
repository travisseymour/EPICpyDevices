from epicpy.epiclib.epiclib import Symbol
import epicpy.epiclib.epiclib.epic_standard_symbols as epic_sym

"""
These are standard names known inside the architecture as symbols
used in perceptual and motor processing. The string-representation
of the system is normally the same as the name with the "_c" suffix removed.
See Epic_standard_names.cpp for the actual definition.
As new names are used in the architecture, they should be added to this set,
and the code should use these variable names instead of explicit strings.
"""

# Standard names for visual and auditory properties:

Detection_c: Symbol = epic_sym.Detection_c
Onset_c: Symbol = epic_sym.Onset_c
Offset_c: Symbol = epic_sym.Offset_c
Start_time_c: Symbol = epic_sym.Start_time_c
End_time_c: Symbol = epic_sym.End_time_c

Eccentricity_c: Symbol = epic_sym.Eccentricity_c
Eccentricity_zone_c: Symbol = epic_sym.Eccentricity_zone_c
Fovea_c: Symbol = epic_sym.Fovea_c
Periphery_c: Symbol = epic_sym.Periphery_c


# Standard names for motor command terms:

Perform_c: Symbol = epic_sym.Perform_c
Prepare_c: Symbol = epic_sym.Prepare_c
Manual_c: Symbol = epic_sym.Manual_c
Keystroke_c: Symbol = epic_sym.Keystroke_c
Hold_c: Symbol = epic_sym.Hold_c
Release_c: Symbol = epic_sym.Release_c
Punch_c: Symbol = epic_sym.Punch_c
Ply_c: Symbol = epic_sym.Ply_c
Point_c: Symbol = epic_sym.Point_c
Click_on_c: Symbol = epic_sym.Click_on_c
Ocular_c: Symbol = epic_sym.Ocular_c
Move_c: Symbol = epic_sym.Move_c
Look_for_c: Symbol = epic_sym.Look_for_c
Vocal_c: Symbol = epic_sym.Vocal_c
Speak_c: Symbol = epic_sym.Speak_c
Subvocalize_c: Symbol = epic_sym.Subvocalize_c
Set_mode_c: Symbol = epic_sym.Set_mode_c
Enable_c: Symbol = epic_sym.Enable_c
Disable_c: Symbol = epic_sym.Disable_c
Centering_c: Symbol = epic_sym.Centering_c
Reflex_c: Symbol = epic_sym.Reflex_c

# Auditory stream names for speech

Overt_c: Symbol = epic_sym.Overt_c
Covert_c: Symbol = epic_sym.Covert_c

# Hand names

Right_c: Symbol = epic_sym.Right_c
Left_c: Symbol = epic_sym.Left_c
Thumb_c: Symbol = epic_sym.Thumb_c
Index_c: Symbol = epic_sym.Index_c
Middle_c: Symbol = epic_sym.Middle_c
Ring_c: Symbol = epic_sym.Ring_c
Little_c: Symbol = epic_sym.Little_c

# Motor processor signal constants

Motor_c: Symbol = epic_sym.Motor_c
Modality_c: Symbol = epic_sym.Modality_c
Processor_c: Symbol = epic_sym.Processor_c
Preparation_c: Symbol = epic_sym.Preparation_c
Execution_c: Symbol = epic_sym.Execution_c
Busy_c: Symbol = epic_sym.Busy_c
Free_c: Symbol = epic_sym.Free_c
Started_c: Symbol = epic_sym.Started_c
Finished_c: Symbol = epic_sym.Finished_c

# Temporal processor

Temporal_c: Symbol = epic_sym.Temporal_c
Ticks_c: Symbol = epic_sym.Ticks_c

# symbols used for standard items in production rules

Tag_c: Symbol = epic_sym.Tag_c
WM_c: Symbol = epic_sym.WM_c

if __name__ == "__main__":
    print(f"{Tag_c=}")
    print(f"{Perform_c=}")
