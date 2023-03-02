"""
TODO
https://github.com/diegoroyo/mitsuba2-transient-nlos/blob/feat-transient/src/librender/scene.cpp

How to extend the Mitsuba 3 scene class? or maybe just create mitransient.Scene which inherits from mitsuba.Scene?

======== TWO FUNCTIONS

/**
    * \brief For non-line of sight scenes, sample a point in the hidden
    *        geometry's surface area. The "hidden geometry" is defined
    *        as any object that does not contain an \ref nloscapturemeter
    *        plugin (i.e. every object but the relay wall)
    *
    * \param ref
    *   A reference point somewhere within the scene
    * 
    * \param sample_
    *   A uniformly distributed 2D vector
    * 
    * \return
    *   Position sampling record
    */
PositionSample3f sample_hidden_geometry_position(const Interaction3f &ref,
                                                    const Point2f &sample_,
                                                    Mask active = true) const {

    MTS_MASK_ARGUMENT(active);

    using ShapePtr = replace_scalar_t<Float, Shape *>;

    Point2f sample(sample_);
    PositionSample3f ps;

    if (likely(!m_hidden_geometries.empty())) {
        if (likely(m_hidden_geometries.size() == 1)) {
            // Fast path if there is only one shape
            ps = m_hidden_geometries[0]->sample_position(ref.time, sample,
                                                         active);
        } else {

            UInt32 index = 0;
            for (Float cpdf : m_hidden_geometries_cpdf) {
                if (sample.x() < cpdf)
                    break;
                index++;
            }

            Float cpdf_before =
                index == 0 ? 0.f : m_hidden_geometries_cpdf[index - 1];
            Float shape_pdf = m_hidden_geometries_cpdf[index] - cpdf_before;

            // Rescale sample.x() to lie in [0,1) again
            sample.x() = (sample.x() - cpdf_before) / shape_pdf;

            ShapePtr shape =
                gather<ShapePtr>(m_hidden_geometries.data(), index, active);

            // Sample a direction towards the emitter
            ps = shape->sample_position(ref.time, sample, active);

            // Account for the discrete probability of sampling this shape
            ps.pdf *= shape_pdf;
        }

        active &= neq(ps.pdf, 0.f);
    } else {
        ps = zero<PositionSample3f>();
    }

    return ps;
}

/**
    * \brief Evaluate the probability density of the  \ref
    *        sample_hidden_geometry_position() technique given an
    *        filled-in \ref PositionSample record.
    *
    * \param ps
    *   A position sampling record, which specifies the query location.
    * 
    * \return
    *    The solid angle density expressed of the sample
    */
Float pdf_hidden_geometry_position(const PositionSample3f &ps,
                                    Mask active = true) const {
    MTS_MASK_ARGUMENT(active);
    using ShapePtr = replace_scalar_t<Float, const Shape *>;

    if (likely(m_hidden_geometries.size() == 1)) {
        // Fast path if there is only one shape
        return m_hidden_geometries[0]->pdf_position(ps, active);
    } else {
        Float shape_pdf = 1.f;
        for (UInt32 i = 1; i < m_hidden_geometries.size(); ++i) {
            if (m_hidden_geometries[i] == ps.object) {
                shape_pdf = m_hidden_geometries_cpdf[i] -
                            m_hidden_geometries_cpdf[i - 1];
                break;
            }
        }

        return reinterpret_array<ShapePtr>(ps.object)->pdf_position(ps,
                                                                    active) *
               shape_pdf;
    }
}


// same as m_shapes, but excluding relay wall objects (i.e. objects
// that are attached to a sensor)
std::vector<ref<Shape>> m_hidden_geometries;
// cumulative PDF of m_hidden_geometries
// m_hidden_geometries_pdf[i] = P(area-weighted random index <= i)
// e.g. if two hidden geometry objects with areas A_1 = 1 and A_2 = 2
// then hidden_geometries_pdf = {0.33f, 1.0f}
std::vector<Float> m_hidden_geometries_cpdf;

======= Extend scene's __init__ function

// NOTE(diego): set scene for sensors, needed by NLOSCaptureSensor
for (Sensor *sensor: m_sensors)
    sensor->set_scene(this);

// NOTE(diego): prepare data for hidden geometry sampling
bool include_relay_wall = true;
TransientSamplingIntegrator<Float, Spectrum> *tsi =
    dynamic_cast<TransientSamplingIntegrator<Float, Spectrum> *>(
        m_integrator.get());
if (tsi) {
    include_relay_wall =
        tsi->hidden_geometry_sampling_includes_relay_wall();
}
Float total_pdf = 0.f;
for (const auto &shape : m_shapes) {
    bool is_relay_wall = false;
    for (const auto &sensor : m_sensors) {
        if (sensor->shape() == shape) {
            is_relay_wall = true;
            break;
        }
    }
    if (!include_relay_wall && is_relay_wall) {
        continue;
    }
    m_hidden_geometries.push_back(shape);
    m_hidden_geometries_cpdf.push_back(total_pdf + shape->surface_area());
    total_pdf += shape->surface_area();
}
for (Float &pdf : m_hidden_geometries_cpdf) {
    pdf /= total_pdf;
}
"""
