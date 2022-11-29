import logging.config
import os
from datetime import datetime
from importlib.resources import read_text
from pathlib import Path

import src.aspire
from src.aspire.config import Config
from src.aspire.exceptions import handle_exception

# version in maj.min.bld format
__version__ = "0.9.1"

# Implements some code that writes out exceptions to 'aspire.err.log'.
# GUY SHARON EDIT
config_txt = '[common]\n# numeric module to use - one of numpy/cupy\nnumeric = numpy\n# fft backend to use - one of pyfftw/scipy/cupy\nfft = pyfftw\n\n[logging]\n# Set log_dir to a relative or absolute directory\n# Default is a subfolder `logs` in your current working directory.\nlog_dir = logs\n# Whether to log any uncaught errors through a sys excepthook\nlog_exceptions = 1\n# More detailed logging config is found in logging.conf\n\n[starfile]\nn_workers = -1\n\n[covar]\ncg_tol = 1e-5\nregularizer = 0.\n\n[mean]\ncg_tol = 1e-5\nregularizer = 0.\n\n[nfft]\nbackends = finufft, cufinufft, pynfft\n'
# config = Config(read_text(aspire, "config.ini"))
config = Config(config_txt)
if config.logging.log_exceptions:
    import sys

    sys.excepthook = handle_exception

# Ensure the log_dir exists
Path(config.logging.log_dir).mkdir(parents=True, exist_ok=True)

##
txt = """[loggers]
keys = root

[handlers]
keys = consoleHandler, fileHandler

[formatters]
keys = simpleFormatter

[logger_root]
level = DEBUG
handlers = consoleHandler, fileHandler

[handler_consoleHandler]
class = StreamHandler
level = INFO
formatter = simpleFormatter
args = (sys.stdout,)

[handler_fileHandler]
class = FileHandler
level = DEBUG
formatter = simpleFormatter
# You may set a custom filename here:
args =(os.path.join('%(log_dir)s', 'aspire-%(dt_stamp)s.log'),)

[formatter_simpleFormatter]
format = %(asctime)s %(levelname)s %(message)s"""
f = open("tmp_logging.conf", "a")
f.write(txt)
f.close()
##

# Generates file name details and opens log file defined in config file.
# The default is to use the current time stamp provided in the dictionary,
#   but that is not required if a user wishes to customize logging config.
logging.config.fileConfig(
    'tmp_logging.conf',
    defaults={
        "dt_stamp": datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f"),
        "log_dir": config.logging.log_dir,
    },
)

##
os.remove("tmp_logging.conf")