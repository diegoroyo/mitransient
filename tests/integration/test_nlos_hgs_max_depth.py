import drjit as dr
import mitsuba as mi
import pytest

mi.set_variant('llvm_ad_rgb')

import mitransient  # noqa: F401


def _integrator(hg_max_depth=None):
    config = {
        'type': 'transient_nlos_path',
        'nlos_hidden_geometry_sampling': True,
    }

    if hg_max_depth is not None:
        config['nlos_hidden_geometry_sampling_max_depth'] = hg_max_depth

    return mi.load_dict(config)


def _apply(integrator, depth, do_hg_sample=True, method_pdf=0.5):
    do_hg_sample, method_pdf = (
        integrator._apply_hg_sampling_depth_limit(
            mi.UInt32(depth),
            mi.Bool(do_hg_sample),
            mi.Float(method_pdf),
        )
    )

    dr.eval(do_hg_sample, method_pdf)

    return bool(do_hg_sample[0]), float(method_pdf[0])


def test_hg_sampling_max_depth_defaults_to_unlimited():
    integrator = _integrator()

    assert integrator.hg_sampling_max_depth == -1

    # The historical behavior must be preserved at arbitrary depth.
    do_hg_sample, method_pdf = _apply(
        integrator,
        depth=10,
        do_hg_sample=True,
        method_pdf=0.5,
    )

    assert do_hg_sample
    assert method_pdf == pytest.approx(0.5)


def test_hg_sampling_max_depth_one_only_allows_depth_zero():
    integrator = _integrator(1)

    do_hg_sample, method_pdf = _apply(
        integrator,
        depth=0,
        do_hg_sample=True,
        method_pdf=0.5,
    )

    assert do_hg_sample
    assert method_pdf == pytest.approx(0.5)

    do_hg_sample, method_pdf = _apply(
        integrator,
        depth=1,
        do_hg_sample=True,
        method_pdf=0.5,
    )

    assert not do_hg_sample

    # Once HGS is disallowed, BSDF sampling is no longer selected with
    # Russian Roulette and therefore has method probability 1.
    assert method_pdf == pytest.approx(1.0)


def test_hg_sampling_max_depth_n_allows_depths_below_n():
    integrator = _integrator(3)

    for depth in (0, 1, 2):
        do_hg_sample, method_pdf = _apply(
            integrator,
            depth=depth,
            do_hg_sample=True,
            method_pdf=0.5,
        )

        assert do_hg_sample
        assert method_pdf == pytest.approx(0.5)

    do_hg_sample, method_pdf = _apply(
        integrator,
        depth=3,
        do_hg_sample=True,
        method_pdf=0.5,
    )

    assert not do_hg_sample
    assert method_pdf == pytest.approx(1.0)


def test_hg_sampling_depth_limit_preserves_bsdf_choice():
    integrator = _integrator(2)

    # At an allowed depth, a BSDF choice made by the existing
    # HGS/BSDF Russian Roulette remains a BSDF choice with p=0.5.
    do_hg_sample, method_pdf = _apply(
        integrator,
        depth=1,
        do_hg_sample=False,
        method_pdf=0.5,
    )

    assert not do_hg_sample
    assert method_pdf == pytest.approx(0.5)

    # Beyond the HGS depth limit, BSDF sampling becomes the only method.
    do_hg_sample, method_pdf = _apply(
        integrator,
        depth=2,
        do_hg_sample=False,
        method_pdf=0.5,
    )

    assert not do_hg_sample
    assert method_pdf == pytest.approx(1.0)


@pytest.mark.parametrize("value", [0, -2])
def test_hg_sampling_max_depth_rejects_invalid_values(value):
    with pytest.raises(
        RuntimeError,
        match="must be -1 or a positive integer",
    ):
        _integrator(value)
