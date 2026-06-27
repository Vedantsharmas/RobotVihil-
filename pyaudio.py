import sys
import pyaudiowpatch

# Alias the pyaudiowpatch module to pyaudio in sys.modules so imports work seamlessly
sys.modules['pyaudio'] = pyaudiowpatch
from pyaudiowpatch import *
