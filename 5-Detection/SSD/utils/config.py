import json
from typing import Tuple, List


class Config:
    """Config stack layers

    - Default config
    - Model default config
    - Load from config file
    - User argument config
    """

    size = (300, 300)

    ssd_attributes = ['feature_map', 'steps', 'sizes', 'aspect_ratios']
    ssd = {
        "aspect_ratios": ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
        "num_priors": 6,
        "variance": (.1, .2),
        "feature_map": (38, 19, 10, 5, 3, 1),
        'sizes': ((30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)),
        "steps": (8, 16, 32, 64, 100, 300),
        "clip": True,

        "warping": False,
        "warping_mode": "sum",
    }

    efficientdet_attributes = ['FPN_D', 'FPN_W', 'CLASS_D', 'OUT']
    efficientdet = {
        "size": (512, 512),
        "FPN_D": 0,
        "FPN_W": 0,
        "CLASS_D": 0,
        "OUT": 0,
    }

    thresh = .3

    conf_thresh = .01

    nms = True
    nms_thresh = .45
    nms_top_k = 200
    variance = .1, .2

    optimizer = {
        "lr": .0001,
        "momentum": .9,
        "weight_decay": 5e-4
    }
    scheduler = {
        "factor": .1,
        "patience": 3,
    }

    def __init__(self, path: str, network: str = None, model: object = None):
        # Update default configs
        for key, value in getattr(self, network.lower(), {}).items():
            self.update(key, value)

        # Update model default configs
        for attribute in getattr(self, f'{network.lower()}_attributes', []):
            self.update(attribute, getattr(model, attribute))

        # Load config files
        if path is not None:
            try:
                with open(path) as f:
                    for key, value in json.load(f).items():
                        self.update(key, value)

            except (FileNotFoundError, RuntimeError) as e:
                print(f'Configfile {path} is not exists or can not open')

    def update(self, key, value):
        if isinstance(getattr(self, key, None), dict):
            getattr(self, key).update(value)
        else:
            setattr(self, key, value)

    def sync(self, arguments: dict):
        for key, value in arguments.items():
            if hasattr(arguments, key):
                setattr(arguments, key, value)
            if key in self.dump.keys():
                self.update(key, value)

    @property
    def dump(self):
        return {
            attr: getattr(self, attr)
            for attr in filter(lambda attr: not attr.startswith('__') and attr != 'dump' and
                                            not callable(getattr(self, attr)), dir(self))
        }
