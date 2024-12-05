# One or more of these individual augmenters were picked from https://github.com/arundo/tsaug
# usage of the repo is present at: https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html
# tsaug is licensed under the Apache License 2.0. See the LICENSE file in the same directory for details.
# Tiny ML Tinyverse aims to improve upon this by adding custom Augmenters
_default_seed = None
from .add_noise import AddNoise
from .convolve import Convolve
from .crop import Crop
from .drift import Drift
from .dropout import Dropout
# from .linear_transform import LinearTransform
from .pool import Pool
from .quantize import Quantize
from .resize import Resize
from .reverse import Reverse
from .time_warp import TimeWarp
