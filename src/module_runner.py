"""
module_runner.py
================
Utility that loads pipeline modules by file path so that numeric
prefixes (01_, 02_, ...) never cause import errors.
"""
import importlib.util, pathlib, sys

_loaded = {}

def load(name: str):
    """Load a .py file from the same directory and return its module object."""
    if name in _loaded:
        return _loaded[name]
    here = pathlib.Path(__file__).parent
    path = here / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod      # register so intra-module imports work
    spec.loader.exec_module(mod)
    _loaded[name] = mod
    return mod
