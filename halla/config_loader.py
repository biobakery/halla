# load configurations from config.yaml
import yaml
from os.path import dirname, abspath, join

yaml_file = join(dirname(abspath(__file__)), 'config.yaml')

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

with open(yaml_file, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as err:
        print(err)

def update_config(attribute, **args):
    vals = getattr(config, attribute)
    for key in args:
        if key not in vals:
            raise KeyError('%s not found in config.%s' % (key, attribute))
        vals[key] = args[key]
    setattr(config, attribute, vals)

config = Struct(**config)