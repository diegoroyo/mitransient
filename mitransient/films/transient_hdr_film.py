import mitsuba as mi

from mitransient.render.transient_block import TransientBlock
import mitsuba


class TransientHDRFilm(mi.Film):
    def __init__(self, props):
        super().__init__(props)

        self.temporal_bins = props.get('temporal_bins', 2048)
        self.bin_width_opl = props.get('bin_width_opl', 0.003)
        self.start_opl = props.get('start_opl', 0)

    def end_opl(self):
        return self.start_opl + self.bin_width_opl * self.temporal_bins

    def add_transient_data(self, spec, distance, wavelengths, active, pos, ray_weight):
        idd = (distance - self.start_opl) / self.bin_width_opl
        coords = mi.Vector3f(pos.x, pos.y, idd)
        mask = (idd >= 0) & (idd < self.temporal_bins)
        self.transient.put(
            coords, wavelengths, spec * ray_weight, mi.Float(1.0), active & mask)

    def prepare(self, aovs):
        # NOTE could be done with mi.load_dict where type='hdrfilm' and the rest of the properties
        props = mi.Properties('hdrfilm')
        props['width'] = self.size().x
        props['height'] = self.size().y
        props['crop_width'] = self.crop_size().x
        props['crop_height'] = self.crop_size().y
        props['crop_offset_x'] = self.crop_offset().x
        props['crop_offset_y'] = self.crop_offset().y
        props['sample_border'] = self.sample_border()
        props['rfilter'] = self.rfilter()
        self.steady = mi.PluginManager.instance().create_object(props)
        self.steady.prepare(aovs)

    def prepare_transient(self, size, channel_count, channel_use_weights, rfilter):
        self.transient = TransientBlock(
            size=size,
            channel_count=channel_count,
            channel_use_weights=channel_use_weights,
            rfilter=rfilter)

    def traverse(self, callback):
        # TODO: all the parameters are set as NonDifferentiable by default
        super().traverse(callback)
        callback.put_parameter('temporal_bins', self.temporal_bins, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('bin_width_opl', self.bin_width_opl, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('start_opl', self.start_opl, mi.ParamFlags.NonDifferentiable)


mi.register_film("transient_hdr_film",
                 lambda props: TransientHDRFilm(props))
