import drjit as dr
import mitsuba as mi


class NLOSSensor(mi.Sensor):
    # This is the abstract base class for NLOS-related sensors in mitransient.
    # This allows to query only this type of sensors while using NLOS-related integrators

    def __init__(self, props: mi.Properties):
        super().__init__(props)
