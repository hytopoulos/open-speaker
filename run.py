#!/usr/bin/env python3

import sys, json
from agent import SpeakerNetAgent
from dotmap import DotMap
import numpy as np

np.set_printoptions(precision=3)

def main(args):
    config_path = args[1]
    config = load_config(config_path, args)

    agent = SpeakerNetAgent(config)
    agent.run()

def load_config(config_path, args):
    with open(config_path, 'r') as f:
        config = DotMap(json.load(f))
        # override the configuration
        for arg in args[2:]:
            key, value = arg.split("=")
            attrs = key.split(".")
            obj = config
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, attrs[-1], type(getattr(obj, attrs[-1]))(value))
    return config

if __name__ == "__main__":
    main(sys.argv)
