def integrator():
    return {
        'type': 'transient_nlos_path',
        'block_size': 1,
        'max_depth': -1,
        'nlos_laser_sampling': True,
        'nlos_hidden_geometry_sampling': True,
        'nlos_hidden_geometry_sampling_includes_relay_wall': False,
        'temporal_filter': 'box',
    }


def Z():
    from mitsuba import ScalarTransform4f as T
    return {
        'type': 'obj',
        'filename': 'examples/nlos/Z.obj',
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [1.0, 1.0, 1.0],
            },
        },
        'to_world': T.translate([0.0, 0.0, 1.0]),
    }


def laser():
    from mitsuba import ScalarTransform4f as T
    return {
        'type': 'projector',
        'to_world': T.translate([-0.5, 0.0, 0.25]),
        'irradiance': {
            'type': 'rgb',
            'value': [1.0, 1.0, 1.0],
        },
        'fov': 0.2,
    }


def sensor(capture_type, *, sx=1, sy=1, spp=5000):
    # TODO add confocal and exhaustive parameters/etc
    return {
        'type': 'nlos_capture_meter',
        'sampler': {
                'type': 'independent',
                'sample_count': spp,
                'seed': 0,
        },
        'account_first_and_last_bounces': False,
        'sensor_origin': [-0.5, 0.0, 0.25],
        'film': {
            'type': 'transient_hdr_film',
            'width': sx,
            'height': sy,
            'temporal_bins': 300,
            'bin_width_opl': 0.006,
            'start_opl': 1.85,
            'rfilter': {
                    'type': 'box',
            },
        },
    }


def relay_wall(sensor):
    return {
        'type': 'rectangle',
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [1.0, 1.0, 1.0],
            },
        },
        'sensor': sensor(),
    }


def test00_Z_single():
    import drjit as dr
    import mitsuba as mi
    mi.set_variant('llvm_ad_rgb')
    import mitransient as mitr
    sx, sy = 4, 2
    relay_wall_obj = mi.load_dict(relay_wall(
        lambda: sensor('single', sx=sx, sy=sy, spp=1)))
    laser_obj = mi.load_dict(laser())
    scene = mi.load_dict(
        {
            'type': 'scene',
            'integrator': integrator(),
            'Z': Z(),
            'laser': laser_obj,
            'relay_wall': relay_wall_obj,
        }
    )

    mitr.nlos.focus_emitter_at_relay_wall_pixel(
        mi.Point2f(sx / 2, sy / 2),
        relay_wall_obj,
        laser_obj)

    transient_integrator = scene.integrator()
    transient_integrator.prepare_transient(scene, sensor=0)

    # Render the scene and develop the data
    data_steady, data_transient = transient_integrator.render(scene)
    # And evaluate the output to launch the corresponding kernel
    dr.eval(data_steady, data_transient)

    # FIXME the data is transposed when comparing steady and transient
    # TAL expects (sx, sy) format for data_transient, does not care about
    # data_steady
    # maybe this is just not a problem
    assert data_steady.shape == (sy, sx, 3)
    assert data_transient.shape == (sx, sy, 300, 3)
