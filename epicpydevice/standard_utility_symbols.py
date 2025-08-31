from epicpy.epiclib.epiclib import Symbol
from epicpy.epiclib.epiclib.standard_utility_symbols import (Default_c, Absent_c, Unknown_c, None_c, Nil_c, Empty_string_c)

"""
These are standard names known inside the architecture as symbols
used in perceptual and motor processing. The string-representation
of the system is normally the same as the name with the "_c" suffix removed.
See Epic_standard_names.cpp for the actual definition.
As new names are used in the architecture, they should be added to this set,
and the code should use these variable names instead of explicit strings.
"""

# Default_c: Symbol = standard_utility_symbols.Default_c
# Absent_c: Symbol = standard_utility_symbols.Absent_c
# Unknown_c: Symbol = standard_utility_symbols.Unknown_c
# None_c: Symbol = standard_utility_symbols.None_c
# Nil_c: Symbol = standard_utility_symbols.Nil_c
# Empty_string_c: Symbol = standard_utility_symbols.Empty_string_c

if __name__ == "__main__":
    print(f"{Default_c=}")
    print(f"{Nil_c=}")
