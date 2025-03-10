from enum import Enum

class InitializerType(Enum):
    """
    InitializerType Enum.

    This enums is used to define the type of weight initialization.
    Attributes:
        ZERO: Zero initialization
        RANDOM_DIST_UNIFORM: Random distribution uniform initialization
        RANDOM_DIST_NORMAL: Random distribution normal initialization
    """
    ZERO = 0
    RANDOM_DIST_UNIFORM = 1
    RANDOM_DIST_NORMAL = 2