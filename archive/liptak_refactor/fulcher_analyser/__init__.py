import pathlib
import sys

# temporarily add this module's directory to PATH
_fulcher_base = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(_fulcher_base))

# module-level imports

# remove unneeded names from namespace
del pathlib, sys

# fulcher_analyser version
__version__ = "0.0.1"
