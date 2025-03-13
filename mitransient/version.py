# mitransient version
__version__ = '1.1.1'

# Mitsuba minimum and maximum compatible versions
__mi_version_min__ = '3.6.0'
__mi_version_max__ = '3.7.0'


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
    mitsuba_supported_min = Version(__mi_version_min__)
    mitsuba_supported_max = Version(__mi_version_max__)

    if mitsuba_version < mitsuba_supported_min:
        raise RuntimeError(
            f'mitransient v{mitransient_version} only supports Mitsuba 3 v{mitsuba_supported_min} to v{mitsuba_supported_max}. '
            f'You are using Mitsuba ({mitsuba_version}). Please upgrade Mitsuba to v{mitsuba_supported_max} (You can use the command `pip install -U mitsuba=={mitsuba_supported_max}`).')
    elif mitsuba_version > mitsuba_supported_max:
        raise RuntimeError(
            f'mitransient v{mitransient_version} only supports Mitsuba 3 v{mitsuba_supported_min} to v{mitsuba_supported_max}. '
            f'You are using Mitsuba ({mitsuba_version}). Please downgrade Mitsuba to v{mitsuba_supported_max} (You can use the command `pip install -U mitsuba=={mitsuba_supported_max}`).')
    return True
