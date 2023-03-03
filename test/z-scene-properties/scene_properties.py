import mitsuba as mi
mi.set_variant('llvm_ad_rgb')  # nopep8

# Extra imports
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt

import mitransient as mitr

scene = mi.load_file(
    '/media/pleiades/vault/projects/202110-nlos-render/mitsuba3-transient/mitsuba3-transient/test/z-scene-properties/nlos_scene.xml')

# integrator = scene.integrator()
# transient_integrator.prepare_transient(kernel_scene, 0)
