# mitransient version
__version__ = '1.3.1'

# Mitsuba minimum and maximum compatible versions
__mi_version_min__ = '3.6.0'
__mi_version_latest__ = '3.9.1'
__mi_version_max__ = '3.10.0'


class Version:
    def __init__(self, string) -> None:
        # Only the leading major.minor.patch numbers are compared, so that
        # strings such as '3.9.1.dev0' or '3.9.1rc1' are also accepted
        import re
        match = re.match(r'^\s*v?(\d+)\.(\d+)\.(\d+)', string)
        if match is None:
            raise RuntimeError(
                f'Version string {string} expected to start with three numbers')
        self.version = tuple(int(x) for x in match.groups())

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
    mitsuba_supported_latest = Version(__mi_version_latest__)
    mitsuba_supported_max = Version(__mi_version_max__)

    if mitsuba_version == Version('3.7.1'):
        mi.Log(mi.LogLevel.Warn,
               f'Mitsuba v{mitsuba_version} has a known issue that causes it to crash when used with mitransient. '
               f'To avoid this, upgrade Mitsuba to v3.8.0 or higher (You can use the command `pip install -U mitsuba==3.8.0`).')

    supported = True
    if mitsuba_version < mitsuba_supported_min:
        supported = False
        mi.Log(mi.LogLevel.Warn,
            f'mitransient v{mitransient_version} only supports Mitsuba 3 at least v{mitsuba_supported_min} and strictly less than v{mitsuba_supported_max}. '
            f'You are using Mitsuba ({mitsuba_version}). Things may not work as expected. Please upgrade Mitsuba to v{mitsuba_supported_latest} (You can use the command `pip install -U mitsuba=={mitsuba_supported_latest}`).')
    elif mitsuba_version >= mitsuba_supported_max:
        supported = False
        mi.Log(mi.LogLevel.Warn,
            f'mitransient v{mitransient_version} only supports Mitsuba 3 at least v{mitsuba_supported_min} and strictly less than v{mitsuba_supported_max}. '
            f'You are using Mitsuba ({mitsuba_version}). Things may not work as expected. Please downgrade Mitsuba to v{mitsuba_supported_latest} (You can use the command `pip install -U mitsuba=={mitsuba_supported_latest}`).')
    return supported
