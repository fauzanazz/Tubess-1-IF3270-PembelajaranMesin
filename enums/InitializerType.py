from enum import Enum

class InitializerType(Enum):
    """
    InitializerType Enum.

    This enums is used to define the type of weight initialization.
    Attributes:
        ZERO: Zero initialization
        RANDOM_DIST_UNIFORM: Random distribution uniform initialization
        RANDOM_DIST_NORMAL: Random distribution normal initialization
        XAVIER : Xavier uniform distribution initialization
        HE : He normal distribution initialization
    """
    ZERO = 0
    RANDOM_DIST_UNIFORM = 1
    RANDOM_DIST_NORMAL = 2
    XAVIER = 3
    HE = 4