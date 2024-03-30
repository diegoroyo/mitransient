# MiTransient version
__version__ = '1.0.0'

# Mitsuba minimum compatible version
__mi_version__ = '3.5.0'


class Version:
    def __init__(self, string) -> None:
        data = string.split('.')
        if len(data) != 3:
            raise RuntimeError(
                f'Version string {string} expected to have three numbers')
        self.version = (data[0], data[1], data[2])

    def __eq__(self, other):
        return self.version == other.version

    def __ne__(self, other):
        return self.version != other.version

    def __ge__(self, other):
        return self.version >= other.version

    def __gt__(self, other):
        return self.version > other.version

    def __le__(self, other):
        return self.version <= other.version

    def __lt__(self, other):
        return self.version < other.version

    def __str__(self) -> str:
        return f'{self.version[0]}.{self.version[1]}.{self.version[2]}'

    def __repr__(self) -> str:
        return self.__str__()


def check_compatibility():
    import os
    os.environ.setdefault('MI_DEFAULT_VARIANT', 'llvm_ad_rgb')

    import mitsuba as mi

    mitransient_version = Version(__version__)
    mitsuba_version = Version(mi.MI_VERSION)
    mitsuba_supported = Version(__mi_version__)

    if mitsuba_version < mitsuba_supported:
        raise RuntimeError(
            f'MiTransient ({mitransient_version}) supports at least Mitsuba ({mitsuba_supported}). Currently installed is Mitsuba ({mitsuba_version}). Please upgrade it.')
    return True
