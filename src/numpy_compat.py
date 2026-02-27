"""
NumPy 2.x Compatibility Patch for DualXDA

This module patches NumPy 2.x to be compatible with scipy's import mechanism.
Import this BEFORE importing any scipy modules.

Usage:
    import numpy_compat  # This patches numpy
    from scipy.sparse import load_npz  # Now this works
"""

import numpy as np
import types
import sys

def patch_numpy():
    """
    Patch NumPy 2.x to add missing modules that scipy expects.
    
    NumPy 2.x removed numpy.strings and numpy.char, but scipy's
    array_api_compat does "from numpy import *" which triggers
    numpy.__getattr__ to try importing these modules.
    
    This function creates dummy modules to prevent the error.
    """
    # Only patch if needed
    if int(np.__version__.split('.')[0]) < 2:
        return  # NumPy 1.x doesn't need patching
    
    # Create numpy.strings if it doesn't exist
    if 'numpy.strings' not in sys.modules:
        strings_module = types.ModuleType('numpy.strings')
        
        # Add basic string functions
        strings_module.add = lambda a, b: str(a) + str(b)
        strings_module.multiply = lambda a, n: str(a) * n
        strings_module.mod = lambda a, values: a % values
        strings_module.capitalize = lambda a: str(a).capitalize()
        strings_module.lower = lambda a: str(a).lower()
        strings_module.upper = lambda a: str(a).upper()
        strings_module.strip = lambda a, chars=None: str(a).strip(chars)
        strings_module.lstrip = lambda a, chars=None: str(a).lstrip(chars)
        strings_module.rstrip = lambda a, chars=None: str(a).rstrip(chars)
        strings_module.split = lambda a, sep=None, maxsplit=-1: str(a).split(sep, maxsplit)
        strings_module.join = lambda sep, seq: sep.join(seq)
        strings_module.replace = lambda a, old, new, count=-1: str(a).replace(old, new, count)
        
        # Register in both numpy and sys.modules
        np.strings = strings_module
        sys.modules['numpy.strings'] = strings_module
    
    # Create numpy.char if it doesn't exist
    if 'numpy.char' not in sys.modules:
        # Reuse strings module for char
        np.char = np.strings
        sys.modules['numpy.char'] = np.strings

# Apply patch immediately when this module is imported
patch_numpy()

# Made with Bob
