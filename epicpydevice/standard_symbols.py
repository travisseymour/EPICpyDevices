from epiclibcpp.epiclib import Symbol, standard_symbols as std_sym


"""
include this file in any C++ code that needs to use Symbols commonly used in models
The actual symbol string value is the same as the name without the "_c" suffix.
Add new values to this set if they are likely to be re-used in either GLEAN or EPIC
Architecture-specific Symbols are defined in <architecture>_standard_symbols
"""

""" Visual objects, properties, and values """

"""
standard visual objects
A "standard" visible object created or maintained by the device should use
one of these as the identity name of the object, and the architecture assumes
the existence of these objects. The actual name is the symbol name without "_name_c"
"""

Cursor_name_c: Symbol = std_sym.Cursor_name_c  # the cursor visual object
Tracking_cursor_name_c: Symbol = (
    std_sym.Tracking_cursor_name_c
)  # the tracking cursor visual object
Mouse_name_c: Symbol = std_sym.Mouse_name_c  # the mouse device itself
Mouse_Left_Button_c: Symbol = std_sym.Mouse_Left_Button_c  # use for single mouse button
Mouse_Middle_Button_c: Symbol = std_sym.Mouse_Middle_Button_c
Mouse_Right_Button_c: Symbol = std_sym.Mouse_Right_Button_c
Keyboard_name_c: Symbol = std_sym.Keyboard_name_c  # the keyboard device as a whole

"""used as a property name - but might be reserved"""
Status_c: Symbol = std_sym.Status_c

"""standard symbols for visual status and changes"""
Visual_c: Symbol = std_sym.Visual_c
Visual_status_c: Symbol = std_sym.Visual_status_c
New_c: Symbol = std_sym.New_c
Visible_c: Symbol = std_sym.Visible_c
Targetable_c: Symbol = std_sym.Targetable_c
Disappearing_c: Symbol = std_sym.Disappearing_c
Reappearing_c: Symbol = std_sym.Reappearing_c
Disappeared_object_c: Symbol = std_sym.Disappeared_object_c
Change_c: Symbol = std_sym.Change_c
Color_changed_c: Symbol = std_sym.Color_changed_c
BgColor_changed_c: Symbol = std_sym.BgColor_changed_c
FgColor_changed_c: Symbol = std_sym.FgColor_changed_c

"""standard visual property names"""
Color_c: Symbol = std_sym.Color_c
Color_Vague_c: Symbol = std_sym.Color_Vague_c
BgColor_c: Symbol = std_sym.BgColor_c
FgColor_c: Symbol = std_sym.FgColor_c
Shape_c: Symbol = std_sym.Shape_c
Size_c: Symbol = std_sym.Size_c  # for actual size
Encoded_size_c: Symbol = std_sym.Encoded_size_c
Label_c: Symbol = std_sym.Label_c
Text_c: Symbol = std_sym.Text_c
Position_c: Symbol = std_sym.Position_c
Vposition_c: Symbol = std_sym.Vposition_c
Hposition_c: Symbol = std_sym.Hposition_c
Location_c: Symbol = std_sym.Location_c
Orientation_c: Symbol = std_sym.Orientation_c
Distance_c: Symbol = std_sym.Distance_c
Depth_c: Symbol = std_sym.Depth_c
Leader_c: Symbol = std_sym.Leader_c  # for radar display leader lines

"""values for other standard visual properties"""
Horizontal_c: Symbol = std_sym.Horizontal_c
Vertical_c: Symbol = std_sym.Vertical_c
Top_c: Symbol = std_sym.Top_c
Bottom_c: Symbol = std_sym.Bottom_c

"""Type_c and standard values"""
Type_c: Symbol = std_sym.Type_c
Screen_c: Symbol = std_sym.Screen_c
Window_c: Symbol = std_sym.Window_c
# Button_c:Symbol = std_sym.Button_c # see shapes
Dialog_c: Symbol = std_sym.Dialog_c
Menu_c: Symbol = std_sym.Menu_c
Menu_item_c: Symbol = std_sym.Menu_item_c
Field_c: Symbol = std_sym.Field_c

"""standard Color property values"""
Aqua_c: Symbol = std_sym.Aqua_c
Black_c: Symbol = std_sym.Black_c
Blue_c: Symbol = std_sym.Blue_c
Brown_c: Symbol = std_sym.Brown_c
Chartreuse_c: Symbol = std_sym.Chartreuse_c
Cyan_c: Symbol = std_sym.Cyan_c
DarkBlue_c: Symbol = std_sym.DarkBlue_c
DarkGray_c: Symbol = std_sym.DarkGray_c
DarkGreen_c: Symbol = std_sym.DarkGreen_c
DarkRed_c: Symbol = std_sym.DarkRed_c
DarkViolet_c: Symbol = std_sym.DarkViolet_c
Gainsboro_c: Symbol = std_sym.Gainsboro_c
Green_c: Symbol = std_sym.Green_c
Gray_c: Symbol = std_sym.Gray_c
Fuchsia_c: Symbol = std_sym.Fuchsia_c
Gold_c: Symbol = std_sym.Gold_c
GoldenRod_c: Symbol = std_sym.GoldenRod_c
LightBlue_c: Symbol = std_sym.LightBlue_c
LightGray_c: Symbol = std_sym.LightGray_c
Magenta_c: Symbol = std_sym.Magenta_c
Maroon_c: Symbol = std_sym.Maroon_c
Navy_c: Symbol = std_sym.Navy_c
Olive_c: Symbol = std_sym.Olive_c
Pink_c: Symbol = std_sym.Pink_c
Purple_c: Symbol = std_sym.Purple_c
Red_c: Symbol = std_sym.Red_c
RoyalBlue_c: Symbol = std_sym.RoyalBlue_c
SlateGray_c: Symbol = std_sym.SlateGray_c
Teal_c: Symbol = std_sym.Teal_c
Turquoise_c: Symbol = std_sym.Turquoise_c
Violet_c: Symbol = std_sym.Violet_c
White_c: Symbol = std_sym.White_c
Yellow_c: Symbol = std_sym.Yellow_c
Vague_c: Symbol = std_sym.Vague_c

"""standard Shape property values"""
Circle_c: Symbol = std_sym.Circle_c
Empty_Circle_c: Symbol = std_sym.Empty_Circle_c
Filled_Circle_c: Symbol = std_sym.Filled_Circle_c
Top_Semicircle_c: Symbol = std_sym.Top_Semicircle_c
Empty_Top_Semicircle_c: Symbol = std_sym.Empty_Top_Semicircle_c
Filled_Top_Semicircle_c: Symbol = std_sym.Filled_Top_Semicircle_c
Rectangle_c: Symbol = std_sym.Rectangle_c
Empty_Rectangle_c: Symbol = std_sym.Empty_Rectangle_c
Filled_Rectangle_c: Symbol = std_sym.Filled_Rectangle_c
Square_c: Symbol = std_sym.Square_c
Empty_Square_c: Symbol = std_sym.Empty_Square_c
Filled_Square_c: Symbol = std_sym.Filled_Square_c
Button_c: Symbol = std_sym.Button_c
Empty_Button_c: Symbol = std_sym.Empty_Button_c
Filled_Button_c: Symbol = std_sym.Filled_Button_c
Triangle_c: Symbol = std_sym.Triangle_c
Empty_Triangle_c: Symbol = std_sym.Empty_Triangle_c
Filled_Triangle_c: Symbol = std_sym.Filled_Triangle_c
Diamond_c: Symbol = std_sym.Diamond_c
Empty_Diamond_c: Symbol = std_sym.Empty_Diamond_c
Filled_Diamond_c: Symbol = std_sym.Filled_Diamond_c
House_c: Symbol = std_sym.House_c
Empty_House_c: Symbol = std_sym.Empty_House_c
Filled_House_c: Symbol = std_sym.Filled_House_c
Inv_House_c: Symbol = std_sym.Inv_House_c
Inv_Empty_House_c: Symbol = std_sym.Inv_Empty_House_c
Inv_Filled_House_c: Symbol = std_sym.Inv_Filled_House_c
Cross_c: Symbol = std_sym.Cross_c
Empty_Cross_c: Symbol = std_sym.Empty_Cross_c
Filled_Cross_c: Symbol = std_sym.Filled_Cross_c
Bar_c: Symbol = std_sym.Bar_c
Empty_Bar_c: Symbol = std_sym.Empty_Bar_c
Filled_Bar_c: Symbol = std_sym.Filled_Bar_c
Clover_c: Symbol = std_sym.Clover_c
Empty_Clover_c: Symbol = std_sym.Empty_Clover_c
Filled_Clover_c: Symbol = std_sym.Filled_Clover_c
Clover3_c: Symbol = std_sym.Clover3_c
Empty_Clover3_c: Symbol = std_sym.Empty_Clover3_c
Filled_Clover3_c: Symbol = std_sym.Filled_Clover3_c
Inv_Clover3_c: Symbol = std_sym.Inv_Clover3_c
Inv_Empty_Clover3_c: Symbol = std_sym.Inv_Empty_Clover3_c
Inv_Filled_Clover3_c: Symbol = std_sym.Inv_Filled_Clover3_c
Heart_c: Symbol = std_sym.Heart_c
Empty_Heart_c: Symbol = std_sym.Empty_Heart_c
Filled_Heart_c: Symbol = std_sym.Filled_Heart_c
Hill_c: Symbol = std_sym.Hill_c
Empty_Hill_c: Symbol = std_sym.Empty_Hill_c
Filled_Hill_c: Symbol = std_sym.Filled_Hill_c
Inv_Hill_c: Symbol = std_sym.Inv_Hill_c
Inv_Empty_Hill_c: Symbol = std_sym.Inv_Empty_Hill_c
Inv_Filled_Hill_c: Symbol = std_sym.Inv_Filled_Hill_c
Cross_Hairs_c: Symbol = std_sym.Cross_Hairs_c
Cursor_Arrow_c: Symbol = std_sym.Cursor_Arrow_c
Up_Arrow_c: Symbol = std_sym.Up_Arrow_c
Down_Arrow_c: Symbol = std_sym.Down_Arrow_c
Left_Arrow_c: Symbol = std_sym.Left_Arrow_c
Right_Arrow_c: Symbol = std_sym.Right_Arrow_c
Line_c: Symbol = std_sym.Line_c
Polygon_c: Symbol = std_sym.Polygon_c
Empty_Polygon_c: Symbol = std_sym.Empty_Polygon_c
Filled_Polygon_c: Symbol = std_sym.Filled_Polygon_c

"""EPICpy: new arrow shapes"""
NN_Arrow_c = Symbol("NN_Arrow")
SS_Arrow_c = Symbol("SS_Arrow")
EE_Arrow_c = Symbol("EE_Arrow")
WW_Arrow_c = Symbol("WW_Arrow")
NE_Arrow_c = Symbol("NE_Arrow")
NW_Arrow_c = Symbol("NW_Arrow")
SE_Arrow_c = Symbol("SE_Arrow")
SW_Arrow_c = Symbol("SW_Arrow")

"""special Shape values"""
North_Leader_c: Symbol = std_sym.North_Leader_c
North_c: Symbol = std_sym.North_c
South_Leader_c: Symbol = std_sym.South_Leader_c
South_c: Symbol = std_sym.South_c
East_Leader_c: Symbol = std_sym.East_Leader_c
East_c: Symbol = std_sym.East_c
West_Leader_c: Symbol = std_sym.West_Leader_c
West_c: Symbol = std_sym.West_c

"""EPICpy: new Directions & Leaders"""
NorthWest_Leader_c = Symbol("NorthWest_Leader")
NorthWest_c = Symbol("NorthWest")
NorthEast_Leader_c = Symbol("NorthEast_Leader")
NorthEast_c = Symbol("NorthEast")
SouthWest_Leader_c = Symbol("SouthWest_Leader")
SouthWest_c = Symbol("SouthWest")
SouthEast_Leader_c = Symbol("SouthEast_Leader")
SouthEast_c = Symbol("SouthEast")

"""special Shape values for visual search targets"""
T000_c: Symbol = std_sym.T000_c
T090_c: Symbol = std_sym.T090_c
T135_c: Symbol = std_sym.T135_c
T180_c: Symbol = std_sym.T180_c
T270_c: Symbol = std_sym.T270_c
L000_c: Symbol = std_sym.L000_c
L090_c: Symbol = std_sym.L090_c
L135_c: Symbol = std_sym.L135_c
L180_c: Symbol = std_sym.L180_c
L270_c: Symbol = std_sym.L270_c
Block_X_c: Symbol = std_sym.Block_X_c
Block_Y_c: Symbol = std_sym.Block_Y_c

"""standard visual property names that have another object name as the value"""
Left_of_c: Symbol = std_sym.Left_of_c
Right_of_c: Symbol = std_sym.Right_of_c
Above_c: Symbol = std_sym.Above_c
Below_c: Symbol = std_sym.Below_c
Inside_c: Symbol = std_sym.Inside_c
In_center_of_c: Symbol = std_sym.In_center_of_c
Outside_c: Symbol = std_sym.Outside_c
In_row_c: Symbol = std_sym.In_row_c
In_col_c: Symbol = std_sym.In_col_c
Placed_on_c: Symbol = std_sym.Placed_on_c
Pointing_to_c: Symbol = std_sym.Pointing_to_c

""" Auditory properties and values """
Auditory_c: Symbol = std_sym.Auditory_c
Auditory_status_c: Symbol = std_sym.Auditory_status_c
Stream_c: Symbol = std_sym.Stream_c
Azimuth_c: Symbol = std_sym.Azimuth_c
Audible_c: Symbol = std_sym.Audible_c
Fading_c: Symbol = std_sym.Fading_c
Pitch_c: Symbol = std_sym.Pitch_c
Loudness_c: Symbol = std_sym.Loudness_c
Timbre_c: Symbol = std_sym.Timbre_c
Present_c: Symbol = std_sym.Present_c
Next_c: Symbol = std_sym.Next_c
Time_stamp_c: Symbol = std_sym.Time_stamp_c
External_c: Symbol = (
    std_sym.External_c
)  # use when distinct localized stream objects unnecessary
Default_physical_stream_c: Symbol = std_sym.Default_physical_stream_c
Default_psychological_stream_c: Symbol = std_sym.Default_psychological_stream_c
Speech_c: Symbol = std_sym.Speech_c
Content_c: Symbol = std_sym.Content_c
Gender_c: Symbol = std_sym.Gender_c
Male_c: Symbol = std_sym.Male_c
Female_c: Symbol = std_sym.Female_c
Speaker_c: Symbol = std_sym.Speaker_c


""" Other standard Symbols """

"""standard signal values"""
Signal_c: Symbol = std_sym.Signal_c
Start_c: Symbol = std_sym.Start_c
Stop_c: Symbol = std_sym.Stop_c
Halt_c: Symbol = std_sym.Halt_c
End_c: Symbol = std_sym.End_c  # For temporal
Increment_c: Symbol = std_sym.Increment_c  # For temporal

if __name__ == "__main__":
    print(f"{Signal_c=}")
    print(f"{Empty_Diamond_c}")
