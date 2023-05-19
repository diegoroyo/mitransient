import mitsuba as mi
mi.set_variant('llvm_rgb')  # nopep8

# Extra imports
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt

import mitransient as mitr

scene = mi.load_file(
    '/media/pleiades/vault/projects/202110-nlos-render/mitsuba3-transient/mitsuba3-transient/test/z-scene-properties/nlos_scene.xml')

transient_integrator = scene.integrator()
transient_integrator.prepare_transient(scene, 0)
# integrator = scene.integrator()
# transient_integrator.prepare_transient(kernel_scene, 0)

img = transient_integrator.render(scene, spp=1_000, seed=50)

img_transient = transient_integrator.transient_block.develop(
    raw=False, gamma=False)

np.save('img.npy', np.array(img))
np.save('img_transient.npy', np.array(img_transient))
