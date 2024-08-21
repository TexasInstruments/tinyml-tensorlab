#################################################################################
# Copyright (c) 2023-2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################

import os
import sys
import yaml
import atexit
import getpass
import logging
import datetime
import pathlib
import platform
import traceback
from time import localtime, strftime
from collections import defaultdict


try:
    import textwrap
    textwrap.indent
except AttributeError:  # undefined function (wasn't added until Python 3.3)
    def indent(text, amount, ch=' '):
        padding = amount * ch
        return ''.join(padding + line for line in text.splitlines(True))
else:
    def indent(text, amount, ch=' '):
        return textwrap.indent(text, amount * ch)


class MyFormatter(logging.Formatter):
    """
    Customising the formatter class.
    overriding the format method of logging.Formatter class
    """
    default_fmt = '%(levelname)7s: %(name)-20s: %(message)s'
    debug_fmt = '%(levelname)7s: %(asctime)-20s: %(name)s: %(message)s'

    def __init__(self, fmt=default_fmt):
        logging.Formatter.__init__(self, fmt, "%H:%M:%S")  # %Y-%m-%d

    def format(self, record):
        format_orig = self._fmt
#        code for indenting new lines. Unable to overwrite original message
#        record.message = record.getMessage()
#        msg_lines = record.message.splitlines()
#        record.message = "\n".join([msg_lines[0], indent("\n".join(msg_lines[1:]), 15)])
        if record.levelno == logging.DEBUG:
            self._fmt = MyFormatter.debug_fmt
        elif record.levelno in [logging.INFO, logging.ERROR, logging.WARN]:
            self._fmt = MyFormatter.default_fmt
        result = logging.Formatter.format(self, record)
        self._fmt = format_orig
        return result


class MsgCounterHandler(logging.StreamHandler):
    """
    Logging handler which keeps an internal count of messages.
    """
    def __init__(self, *args, **kwargs):
        super(MsgCounterHandler, self).__init__(*args, **kwargs)
        self._counts = defaultdict(lambda: defaultdict(int))

    def emit(self, record):
        record.count = self._counts[record.name][record.levelname]
        super(MsgCounterHandler, self).emit(record)
        self._counts[record.name][record.levelname] += 1
        # print(record)
        # print(self._counts)


def add_log_file_handlers(logger, base_log_filename=None, extra_log_files=[]):
    """
    base_log_filename: Used as a default name if a directory is given in extra_log_files
    extra_log_files format: List containing dictionary or file_path:
        [{"path": "/tmp/log_file1", "level": "DEBUG"}, "/tmp/log_file2"]
    """
    formatter = MyFormatter()
    try:  # For extra log files in different locations
        for log_info in extra_log_files:
            log_level = None
            if type(log_info) is dict:
                if "path" in log_info:
                    log_path = log_info["path"]
                else:
                    logger.error("Key 'path' not given in extra_log_files")
                    continue
                if "level" in log_info:
                    try:
                        if log_info["level"] in ["INFO", "DEBUG", "WARN", "WARNING",
                                                 "ERROR", "CRITICAL"]:
                            log_level = getattr(logging, log_info["level"])
                    except AttributeError:
                        pass
            else:
                log_path = log_info
            if os.path.exists(log_path):
                if os.path.isfile(log_path):
                    log_file = log_path
                elif os.path.isdir(log_path):
                    if base_log_filename is None:
                        logger.warning("No base_log_filename provided. "
                                       "Skipping extra log!")
                        continue
                    else:
                        log_file = os.path.join(log_path, base_log_filename)
            else:
                log_file = log_path
                log_dir = os.path.realpath(os.path.dirname(log_path))
                pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
            f2_handler = logging.FileHandler(log_file, mode="w")
            f2_handler.setFormatter(formatter)
            if log_level is not None:
                f2_handler.setLevel(log_level)
            logger.addHandler(f2_handler)  # write to file
    except Exception:
        logger.info("Unable to add extra logs: {}".format(extra_log_files))


def Logger(log_file=None, DEBUG=False, name="root", extra_log_files=[],
           append_log=False, console_log=True):
    """
    Function to create a logger for logging
    extra_log_files format: List containing dictionary or file_path:
        [{"path": "/tmp/log_file1", "level": "DEBUG"}, "/tmp/log_file2"]
    """
    logger = logging.getLogger(name)
    logger.handlers = []
    formatter = MyFormatter()
    if console_log:
        c_handler = MsgCounterHandler()
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)  # write to console
    if log_file:
        log_dir = os.path.realpath(os.path.dirname(log_file))
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        if append_log:
            f_handler = logging.FileHandler(log_file)
        else:
            f_handler = logging.FileHandler(log_file, mode="w")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)  # write to file
        add_log_file_handlers(logger, os.path.basename(log_file), extra_log_files)
    if DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger


def command_display(lis=None, debug=False, exit_on_error=True, in_logger=None,
                    log_name="root", extra_log_files=[]):
    """
    This function
    1. Requires a compulsory "lis" argument for filename of output.
    2. Initializes a logger and logs to specified file.
    3. Displays the inputs given to the program.
    3. Returns the logger which can be used for further logging into the file
    """
    def get_log_count(log_level):
        logger = logging.getLogger(log_name)
        counts = logger.handlers[0]._counts
        return sum([log[log_level] for log in counts.values()])

    def printlog(start_time):
        # Try to access the last exception that occured. If an exception
        # occured & was left nhandled, sys.last_* will contain such information
        try:
            logger.error("Exception occured during execution\n"+indent("".join(traceback.
                         format_exception(sys.last_type, sys.last_value, sys.last_traceback)), 4))
        except AttributeError:
            pass
        totalTime = datetime.datetime.now() - start_time
        error_count = get_log_count("ERROR")
        warning_count = get_log_count("WARNING")
        exitInfo = "\n  Execution status:\n"
        exitInfo += "{:6} Errors detected.\n".format(error_count)
        exitInfo += "{:6} Warnings detected.\n".format(warning_count)
        exitInfo += "\n  Total execution time : {}\n".format(totalTime)
        logger.info(exitInfo)
        if (error_count > 0) and exit_on_error:
            sys.exit
    logger = in_logger or Logger(lis, debug, log_name, extra_log_files)
    start_time = datetime.datetime.now()
    cur_time = strftime("%Y-%m-%d %H:%M:%S", localtime())
#    printStr = "Script location / Command line switch / prompt values:\n\n"
#    for arg in sys.argv[:-1]:
#        if not re.match(r"\s*-", arg):
#            printStr += " {} \\\n".format(arg)
#        else:
#            printStr += "  {}".format(arg)
#    printStr += " {}\n".format(sys.argv[-1])
#    logger.info(printStr)
    infoStr = "\n  Execution information:\n"
    infoStr += "  Date/Time of run : {}\n".format(cur_time)
    infoStr += "  Node name        : {}\n".format(platform.node())
    infoStr += "  User login       : {}\n".format(getpass.getuser())
    infoStr += "  Operating System : {}\n".format(platform.platform())
    infoStr += "  Python location  : {}\n".format(sys.executable)
    logger.info(infoStr)
    atexit.register(printlog, start_time)
    logger.debug("Logger is initialized")
    return logger


def read_yaml_file(my_file, log_name='root.utils.RYML'):
    """
    This Module reads yaml data into a dictionary and returns it
    """
    log = logging.getLogger(log_name)
    log.debug("###   call to read YAML file into a dict   ###")
    log.info("Reading YAML: {}".format(my_file))
    # check_path(my_file)
    if(check_path(my_file) is None):
        log.error("file/dir path does not exist : {}".format(str(my_file)))
        return
    with open(my_file, 'r') as fp:
        read_data = yaml.load(fp, Loader=yaml.CLoader)
    return read_data


def check_path(path, log_name="root.utils.CKP"):
    """
    Function to check whether the path specified exists.
    Returns the complete path if present.
    Else exits.
    Quiet Mode:
    If logs should not be logged to the root logger, then pass log_name as ""
    """
    logger = logging.getLogger(log_name)
    if path is not None and os.path.exists(path):
        return os.path.abspath(path)
    logger.info("file/dir path {path} not proper".format(path=path))
    return None


def path_arg(path):
    """
    This function checks the existance of the path, if not exist then exits the
    system else return the abspath of path passed
    """
    if not os.path.exists(path):
        raise ValueError("file/dir path does not exist : {}".format(str(path)))
    else:
        return os.path.abspath(path)
    return


def create_dir(dir_path, log_name="root.utils.CDR", propagate_log=True):
    """
    Function to check whether the directory specified exists.
    If not creates the directory.
    Quiet Mode:
    If logs should not be logged to the root logger, then pass log_name as ""
    """
    logger = logging.getLogger(log_name)
    logger.propagate = propagate_log
    if os.path.isdir(dir_path):
        logger.info("directory exists: {dir_path}".format(dir_path=dir_path))
        return
    logger.info("Creating directory: {dir_path}".format(dir_path=dir_path))
    os.makedirs(dir_path)
