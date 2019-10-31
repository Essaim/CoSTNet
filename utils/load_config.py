import json

configpath = "config.json"

with open(configpath) as File:
    config = json.load(configpath)

def get_config(name, defaultValue = None):
    try:
        return config[name]
    except KeyError:
        print(f"invalid keyvalue{name}")
        return defaultValue