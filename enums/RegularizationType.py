from enum import Enum

class RegularizationType(Enum):
    L1 = 1
    L2 = 2
    RMS = 3
    ELASTIC_NET = 4
    NONE = 5