import pytest
import drjit as dr
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")

import mitransient  # noqa: F401


def integrator():
    return {
        "type": "transient_nlos_path",
        "max_depth": -1,
        "nlos_laser_sampling": False,
        "nlos_hidden_geometry_sampling": True,
        "nlos_hidden_geometry_sampling_includes_relay_wall": False,
        "temporal_filter": "box",
    }


def sensor():
    return {
        "type": "nlos_capture_meter",
        "sampler": {
            "type": "independent",
            "sample_count": 1,
            "seed": 0,
        },
        "sensor_origin": [-0.5, 0.0, 0.25],
        "film": {
            "type": "transient_hdr_film",
            "width": 1,
            "height": 1,
            "temporal_bins": 8,
            "bin_width_opl": 0.1,
            "start_opl": 0.0,
            "rfilter": {"type": "box"},
        },
    }


def relay_wall():
    return {
        "type": "rectangle",
        "bsdf": {
            "type": "diffuse",
            "reflectance": {
                "type": "rgb",
                "value": [1.0, 1.0, 1.0],
            },
        },
        "sensor": sensor(),
    }


def hidden_shape(center_x, scale):
    T = mi.ScalarTransform4f
    return {
        "type": "rectangle",
        "to_world": (
            T.translate([center_x, 0.0, 1.0])
            @ T.scale([scale, scale, scale])
        ),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {
                "type": "rgb",
                "value": [0.5, 0.5, 0.5],
            },
        },
    }


def emitter():
    T = mi.ScalarTransform4f
    return {
        "type": "projector",
        "to_world": T.translate([-0.5, 0.0, 0.25]),
        "irradiance": {
            "type": "rgb",
            "value": [1.0, 1.0, 1.0],
        },
        "fov": 20.0,
    }


def make_scene(order):
    scene_dict = {
        "type": "scene",
        "integrator": integrator(),
        "relay_wall": relay_wall(),
        "laser": emitter(),
    }

    objects = {
        "shape_a": hidden_shape(-2.0, 0.5),
        "shape_b": hidden_shape(+2.0, 1.0),
    }

    for name in order:
        scene_dict[name] = objects[name]

    return mi.load_dict(
        scene_dict,
        parallel=False,
        optimize=False,
    )


def sample_hidden_geometry_position(order):
    scene = make_scene(order)
    integrator = scene.integrator()

    integrator.prepare(
        scene=scene,
        sensor=0,
        seed=mi.UInt32(0),
        spp=1,
        aovs=[],
    )

    ref = dr.zeros(mi.Interaction3f)
    sample2 = mi.Point2f(
        mi.Float(0.10),
        mi.Float(0.50),
    )

    ps = integrator._sample_hidden_geometry_position(
        ref,
        scene,
        sample2,
        mi.Bool(True),
    )

    dr.eval(ps.p, ps.pdf)

    p = (
        float(ps.p.x[0]),
        float(ps.p.y[0]),
        float(ps.p.z[0]),
    )
    pdf = float(ps.pdf[0])

    return p, pdf


def test_hidden_geometry_sampling_is_independent_of_scene_shape_order():
    p_ab, pdf_ab = sample_hidden_geometry_position(["shape_a", "shape_b"])
    p_ba, pdf_ba = sample_hidden_geometry_position(["shape_b", "shape_a"])

    assert pdf_ab == pytest.approx(pdf_ba)
    assert p_ab == pytest.approx(p_ba), (
        "Hidden geometry sampling depends on scene.shapes() ordering. "
        f"sample(order=['shape_a','shape_b'])={list(p_ab)}, "
        f"sample(order=['shape_b','shape_a'])={list(p_ba)}"
    )
