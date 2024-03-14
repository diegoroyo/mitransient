import drjit as dr
import mitsuba as mi


class NLOSSensor(mi.Sensor):
    """
    This sensor is just the base class for NLOS related sensors.
    This allows to query only this type of sensors while using NLOS-related integrators
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
