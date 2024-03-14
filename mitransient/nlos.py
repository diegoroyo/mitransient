import mitsuba as mi
import drjit as dr

def focus_emitter_relay_wall(pixel, relay_wall, emitter):
    # First compute uv coordinates of the sensor
    sensor = relay_wall.sensor()
    film_size = sensor.film().size()
    uv = mi.Point2f(
        pixel.x / film_size.x,
        pixel.y / film_size.y
    )

    # Second sample the position in the relay wall
    target = relay_wall.sample_position(0.0, uv, True).p

    # Update emitter to_world transform to focus on that point
    emitter_origin = emitter.world_transform().translation()
    emitter_params = mi.traverse(emitter)
    emitter_params['to_world'] = mi.Transform4f.look_at(
            origin=emitter_origin,
            target=target,
            up=mi.Vector3f(0, 1, 0)
    )
    emitter_params.update()

    # If the sensor cares about the emitter distance, update it
    sensor_params = mi.traverse(sensor)
    if 'laser_bounce_opl' in sensor_params:
        sensor_params['laser_bounce_opl'] = dr.norm(target - emitter_origin)
    sensor_params.update()
