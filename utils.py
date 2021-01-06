# -*- coding: utf-8 -*-
"""
Miscellaneous functions.
"""

import datetime
import platform
import sys

__author__ = "Gilbert Peralta"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "9 July 2020"

TERMINAL_WIDTH = 80

class Colors:
    """
    Class for colored texts in terminal. Available attributes are:

    ----------      ----------
    Attributes      Value
    ----------      ----------
    GREEN           '\033[92m'
    BLUE            '\033[96m'
    WARNING         '\033[93m'
    FAIL            '\033[91m'
    ENDC            '\033[0m'
    BOLD            '\033[1m'
    UNDERLINE       '\033[4m'
    """
    GREEN = '\033[92m'
    BLUE = '\033[96m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_start_implementation():
    """
    Prints machine platform, python version and start datetime of execution.
    """
    print(Colors.BOLD + Colors.GREEN + "*"*TERMINAL_WIDTH + "\n")
    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start of Run: " + start + "\n")

    string = ("PYTHON VERSION: {} \nPLATFORM: {} \nPROCESSOR: {}"
        + "\nVERSION: {} \nMAC VERSION: {}")
    print(string.format(sys.version, platform.platform(),
        platform.uname()[5], platform.version()[:60]
        + "\n" + platform.version()[60:], platform.mac_ver()) + "\n"
        + Colors.ENDC)


def print_end_implementation():
    """
    Prints end datetime of execution.
    """
    end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(Colors.BOLD + Colors.GREEN + "\nEnd of Run: " + end + "\n")
    print("*"*TERMINAL_WIDTH + '\n' + Colors.ENDC)


def print_line():
    """
    Prints a line on the python shell or terminal.
    """
    print(Colors.UNDERLINE + " "*TERMINAL_WIDTH + Colors.ENDC)
