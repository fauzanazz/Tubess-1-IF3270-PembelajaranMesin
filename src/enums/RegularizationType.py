from enum import Enum

class RegularizationType(Enum):
    L1 = 1
    L2 = 2
    ELASTIC_NET = 3
    NONE = 4