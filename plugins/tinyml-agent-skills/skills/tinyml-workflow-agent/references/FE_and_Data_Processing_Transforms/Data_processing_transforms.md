This files lays out the function definitions for the complete list of data processing transformations. Can be used to understand what each data processing transform does to the input data. This information is crucial for evaluating whether or not a particular transform is suited for the input data.

# ==================== Data Processing Transforms ====================

    def _transform_downsample(self, x_temp):
        """Downsample the dataset by a factor (sampling_rate/new_sr)."""
        return basic_transforms.Downsample(x_temp, self.sampling_rate, self.new_sr)

    def _transform_simple_window(self, x_temp):
        """Form windows of size frame_size from the dataset."""
        stride_window = int(self.frame_size * self.stride_size)
        if stride_window == 0:
            self.logger.warning("Stride Window size calculated is 0. Defaulting the value to Sample Window i.e. no overlap.")
            stride_window = self.frame_size
        if not hasattr(self, 'keep_short_tails'):
            self.keep_short_tails = False
        x_temp = np.array(basic_transforms.SimpleWindow(x_temp, self.frame_size, stride_window, self.keep_short_tails))
        if len(self.feat_ext_transform) > 0:
            x_temp = x_temp.reshape(-1, self.variables)
        return x_temp

    def _transform_gain(self, array, label):
        """Apply a gain factor to the array."""
        try:
            gain_variations = literal_eval(self.gain_variations)
            if label in gain_variations.keys():
                gain = random.uniform(gain_variations[label][0], gain_variations[label][1])
                array = array * gain
            else:
                self.logger.warning(f"Dataset class:{label} not found in input gain variations: {gain_variations.keys()}")
        except ValueError as e:
            self.logger.warning(f"Gain Variations couldn't be applied: {e}")
        return array