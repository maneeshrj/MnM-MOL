"""
Miscellaneous helper functions and classes.
"""
import os
import json
import torch
from types import SimpleNamespace


def writeLog(string, fname, mute=False):
    with open(fname, "a", encoding='utf8') as log: # append mode 
        log.write(string+"\n")
    if not mute: print(string)
    

def save_checkpoint(states, path, filename='model_best.pth.tar'):
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)
    
    
class TrainSettings(dict):
    def __init__(self, fname):
        with open(fname, "r") as settingsf:
            mapping = json.load(settingsf)
            self.__dict__.update(mapping)
            print(f"Settings loaded from {fname}")        
            
    def write_json(self, writepath="settings.json"):
        with open(writepath, "w") as settingsf:
            settingsf.write(json.dumps(self.__dict__, indent=4))
            print(f"Settings saved to {writepath}")
            