from enum import Enum

class Facelabel(Enum):
    mask=0
    without_mask=1

class PeoplenetLabel(Enum):
    person=0
    bag=1
    face=2