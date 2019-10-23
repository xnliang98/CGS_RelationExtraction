"""
Helper functions.
"""
import os
import subprocess
import json
import argparse

# file and path operators
def check_dir(d):
    if not os.path.exists(d):
        print("Directory {} does not exist. Exit.".format(d))
        exit(1)

def check_files(files):
    for f in files:
        if f is not None and not os.path.exists(f):
            print("File {} does not exist. Exit.".format(f))
            exit(1)

def ensure_dir(d, verbose=True):
    """ check dir, if not exist, creat it. """
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; Creating...".format(d))
        os.makedirs(d)

# config file 
def save_config(config, path, verbose=True):
    with open(path, 'w') as fout:
        json.dump(config, fout, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config

def load_config(path, verbose=True):
    with open(path, 'r') as fin:
        config = json.load(f)
    if verbose:
        print("Config loaded from {}".format(path))
    return config

def print(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, v)
    print("\n" + info + '\n')
    return None

# log file
class FileLogger(object):
    """
    A logger that opens a file to output log info.
    """
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            # remove existed file
            os.remove(filename)
        if header is not None:
            with open(filename, 'w') as fout:
                print(header, file=fout)
    
    def log(self, message):
        with open(self.filename, 'a') as fout:
            print(message, file=fout)