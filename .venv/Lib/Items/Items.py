import sys
import cv2 as cv
import numpy as np
import math
from enum import Enum

from numpy import dtype
from cv2 import Mat
from numpy import ndarray


class Itemtype(Enum):
    FORK = "FORK"
    KNIFE = "KNIFE"
    PAN = "PAN"
    SPOON = "SPOON"


class Item:
    def __init__(self, image:Mat | ndarray['any', 'dtype'] | ndarray, label:Itemtype):
        self.image = image
        self.label = label
