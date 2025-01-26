"""
Utility function module
"""

import os
import psutil
import sys
import time
import warnings

def ANSI(color:str, text:str):
    """
    Colour output text using ANSI escape codes

    Parameters
    ----------
    color : str
        Text color. Available options include 'R', 'G', 'Y', 'B', 'M', 'C'
    text : str
        Text to be coloured

    Returns
    -------
    ctext : str
        Coloured text string
    """

    c_list = {'R': '91', 'G': '92', 'Y': '93', 'B': '94', 'M': '95', 'C': '96'}

    assert color in c_list.keys(), "Incorrect colour input. " +\
        "Available options are " +\
        ", ".join(["'" + ANSI(c, "%s"%c) + "'" for c in c_list.keys()]) + " only. \n"
    
    ctext = "\033[%sm"%c_list[color] + text + "\033[00m"

    return ctext

def cpu_priority():
    """
    Sets the CPU priority for Python to low
    """

    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    return None

def time_elapsed(func):
    """
    Decorator function that keeps track of the runtime
    """

    def wrapper(*args, **kwargs):
        t0 = time.time()
        func(*args, **kwargs)
        t1 = time.time()

        dt = t1 - t0
        hh = dt//3600
        mm = (dt - hh*3600)//60
        ss = dt - hh*3600 - mm*60
        print("Total Time Elapsed: %02d:%02d:%02d \n"%(hh, mm, ss))

        return dt
    
    return wrapper

def update_progress(progress:float, label="Progress"):
    """
    Creates a progress bar
    
    Parameters
    ----------
    progress : float
        Calculation progress as a decimal between 0 and 1
    label : str, default "Progress"
        Text to be displayed next to the progress bar
    """

    barlength = 50
    status = ''
    block = int(round(barlength*progress))
    filled = ANSI('G', "\u25A0")*block + '-'*(barlength-block)
    text = "\r%s: [%s] %.1f%% %s"%(label, filled, progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

    return None    

def warning_output(msg, cat, fname, lineno, file=None, line=None):
    """
    Modifies the warning output
    """

    return "%s:%s: %s: %s"%(fname, lineno, cat.__name__, msg)


