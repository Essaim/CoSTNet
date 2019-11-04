import json
import os

absPath = os.path.join(os.path.dirname(__file__), "config.json")

with open(absPath) as File:

    config = json.load(File)

def get_config(name, defaultValue = None):
    try:
        return config[name]
    except KeyError:
        print(f"invalid keyvalue{name}")
        return defaultValue