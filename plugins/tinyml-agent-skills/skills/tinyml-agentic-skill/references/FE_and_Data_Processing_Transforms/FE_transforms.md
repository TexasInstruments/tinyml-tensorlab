This files lays out the function definitions for the complete list of feature extraction transformations. Can be used to understand what each Feature Extraction transform does to the input data. This information is crucial for evaluating whether or not a particular transform is suited for the input data.



# ==================== Feature Extraction Transforms ====================

def _transform_raw(self, wave_frame):
    result_wave = wave_frame
    mean_of_wave = np.sum(result_wave) / len(result_wave)
    if self.dc_remove:
        result_wave = result_wave - mean_of_wave
    return wave_frame, result_wave

def _transform_windowing(self, wave_frame):
    result_wave = wave_frame * np.hanning(self.frame_size)
    return wave_frame, result_wave

def _transform_fft(self, wave_frame):
    result_wave = np.fft.fft(wave_frame)
    return wave_frame, result_wave

def _transform_binning(self, wave_frame):
    result_wave = np.empty((self.feature_size_per_frame))
    bin_size = self.bin_size
    bin_offset = self.min_bin if self.min_bin else 0

    for idx in range(self.feature_size_per_frame):
        idx1 = bin_offset + (idx) * bin_size
        sum_of_bin = np.sum(wave_frame[idx1:idx1 + bin_size])
        result_wave[idx] = sum_of_bin
        result_wave[idx] /= bin_size if self.normalize_bin else 1
    return wave_frame, result_wave

def _transform_concatenation(self, features_of_frame_per_ax, index_frame):
    result_wave = np.array(features_of_frame_per_ax[index_frame - self.num_frame_concat + 1: index_frame + 1]).flatten()
    return features_of_frame_per_ax, result_wave

def _transform_pos_half_fft(self, wave_frame):
    """Takes the DC + min_bin samples of the fft."""
    idx = self.frame_size // 2
    idx += self.min_bin if self.min_bin else 1
    result_wave = wave_frame[:idx]
    return wave_frame, result_wave

def _transform_absolute(self, wave_frame):
    """Converts the complex value to real values."""
    result_wave = abs(wave_frame)
    return wave_frame, result_wave

def _transform_haar(self, wave_frame):
    result_wave = haar_forward(wave_frame)
    return wave_frame, result_wave

def _transform_hadamard(self, wave_frame):
    result_wave = hadamard_forward_vectorized(wave_frame)
    return wave_frame, result_wave

def _transform_normalize(self, wave_frame):
    result_wave = wave_frame / self.frame_size
    return wave_frame, result_wave

def _transform_roundoff(self, wave_frame):
    result_wave = np.round((np.array(wave_frame)))
    return wave_frame, result_wave

def _transform_log(self, wave_frame):
    if self.log_base == 'e':
        result_wave = self.log_mul * np.log(self.log_threshold + wave_frame)
    else:
        try:
            base = float(self.log_base)
            result_wave = self.log_mul * np.log(self.log_threshold + wave_frame) / np.log(base)
        except (ValueError, TypeError):
            self.logger.warning("log_base can't be converted to float, default to base 10")
            result_wave = self.log_mul * np.log10(self.log_threshold + wave_frame)
    return wave_frame, result_wave

def _transform_to_q15(self, wave_frame):
    def q15sat(x):
        if x > 0x7FFF:
            return np.int16(0x7FFF)
        elif x < -0x8000:
            return np.int16(-0x8000)
        else:
            return np.int16(x)
    q15satV = np.vectorize(q15sat)
    def toQ15(x):
        return q15satV(np.round(np.array(x) * (1 << 15)))
    result_wave = toQ15(wave_frame)
    return wave_frame, result_wave

def _transform_binning_q15(self, wave_frame):
    result_wave = np.empty((self.feature_size_per_frame))
    bin_size = self.bin_size
    bin_offset = self.min_bin if self.min_bin else 0
    for idx in range(self.feature_size_per_frame):
        idx1 = bin_offset + (idx) * bin_size
        sum_of_bin = np.sum(wave_frame[idx1:idx1 + bin_size])
        avg = sum_of_bin // (bin_size if self.normalize_bin else 1)
        result_wave[idx] = np.clip(avg, -32768, 32767).astype(np.int16)
    return wave_frame, result_wave

def _transform_fft_q15(self, wave_frame):
    inst = dsp.arm_rfft_instance_q15()
    dsp.arm_rfft_init_q15(inst, self.frame_size, 0, 1)
    result_wave = dsp.arm_rfft_q15(inst, wave_frame)
    return wave_frame, result_wave

def _transform_q15_scale(self, wave_frame):
    result_wave = dsp.arm_shift_q15(wave_frame, self.q15_scale_factor)
    return wave_frame, result_wave

def _transform_q15_cmplx_mag(self, wave_frame):
    result_wave = dsp.arm_cmplx_mag_q15(wave_frame[:self.frame_size + 2])
    return wave_frame, result_wave

def _transform_kurtosis(self, wave_frame):
    wave_frame = np.array(wave_frame, dtype=np.float32)
    padded_frame = np.pad(wave_frame, (4, 3), mode='constant', constant_values=0)
    result_wave = []
    stride_window = int(self.frame_size * self.stride_size)
    window_length = len(padded_frame) - (self.window_count * stride_window)
    for i in range(self.window_count):
        start = i * stride_window
        end = start + window_length
        if end > len(padded_frame):
            break
        window = padded_frame[start:end]
        window_kurtosis = []
        chunk_count = window_length // self.chunk_size
        for j in range(chunk_count):
            chunk_start = j * self.chunk_size
            chunk_end = chunk_start + self.chunk_size
            chunk = window[chunk_start:chunk_end]
            eps = 1e-12
            if np.nanstd(chunk) < eps:
                chunk_kurtosis = 0.0
            else:
                chunk_kurtosis = kurtosis(chunk, fisher=True, bias=False)
            window_kurtosis.append(chunk_kurtosis)
        result_wave.append(window_kurtosis)
    result_wave = np.array(result_wave)
    return wave_frame, result_wave

def _transform_slope_changes(self, wave_frame):
    first_derivative = np.diff(wave_frame)
    result_wave = np.sum(np.diff(np.sign(first_derivative)) != 0) / len(first_derivative)
    return wave_frame, result_wave

def _transform_slope_changes_fixed_point(self, wave_frame):
    first_derivative = np.diff(wave_frame)
    total_slope = 0
    currSlope = np.sign(first_derivative[0])
    threshold = 1 
    for ele in first_derivative[1:]:
        if abs(ele) > threshold:
            if np.sign(ele) != currSlope:
                currSlope = np.sign(ele)
                total_slope += 1
    result_wave = total_slope * 1000    # Scaling values by 1000
    return wave_frame, result_wave

def _transform_zero_crossing_rate_fixed_point(self, wave_frame):
    result_wave = np.sum(np.diff(np.sign(wave_frame)) != 0)
    result_wave = result_wave * 1000     # Scaling values by 1000   
    return wave_frame, result_wave

def _transform_zero_crossing_rate(self, wave_frame):
    result_wave = np.sum(np.diff(np.sign(wave_frame)) != 0) / len(wave_frame)
    return wave_frame, result_wave

def _transform_dominant_frequency_fixed_point(self, wave_frame):
    window = wave_frame * 128
    # Symmetrically mirror the window to 128 samples
    _, mirrored_window = self._transform_symmetric_mirror(window)
    num_bins = self.fft_size// 2 + 1 
    # Perform FFT and take absolute values               
    inst = dsp.arm_rfft_instance_q15()
    dsp.arm_rfft_init_q15(inst, self.fft_size,0,1)         # forward FFT
    fft_res = dsp.arm_rfft_q15(inst, mirrored_window)
    mag_res = dsp.arm_cmplx_mag_q15(fft_res)[:num_bins]
    mag_res = mag_res * 128
    mag_res[0] = 0        # DC  Remove
    freqIndex1 = 0
    freqIndex2 = 0
    for i in range (1,num_bins):
        if (mag_res[freqIndex1] < mag_res[i]):
            freqIndex2 = freqIndex1
            freqIndex1 = i
        elif (mag_res[freqIndex2] < mag_res[i]):
            freqIndex2 = i
    maxFreq1 = (((freqIndex1) * float(self.sampling_rate) // (2*num_bins))) * 1000       # Scaling values by 1000
    maxFreq2 = (((freqIndex2) * float(self.sampling_rate) // (2*num_bins))) * 1000       # Scaling values by 1000
    result_wave = [maxFreq1,maxFreq2]
    return wave_frame, result_wave 

def _transform_dominant_frequency(self, wave_frame):
    num_bins = self.fft_size // 2 + 1
    fft_result = np.abs(np.fft.fft(wave_frame, n=self.fft_size))[:num_bins]
    freq_bins = np.linspace(0, float(self.sampling_rate) / 2.0, int(num_bins))
    sorted_indices = np.argsort(fft_result[1:])[-2:] + 1
    result_wave = freq_bins[sorted_indices]
    return wave_frame, result_wave

def _transform_spectral_entropy(self, wave_frame):
    num_bins = self.fft_size // 2 + 1
    fft_result = np.abs(np.fft.fft(wave_frame, n=self.fft_size))[:num_bins]
    power_spectrum = fft_result ** 2
    psd_norm = power_spectrum / np.sum(power_spectrum)
    result_wave = entropy(psd_norm)
    return wave_frame, result_wave

def _transform_symmetric_mirror(self, wave_frame):
    if len(wave_frame) == 0:
        result_wave = np.zeros(self.fft_size, dtype=np.float32)
    else:
        result_wave = np.tile(wave_frame, self.fft_size // len(wave_frame) + 1)
        result_wave = result_wave[:self.fft_size]
    return wave_frame, result_wave

def _transform_pir_feature_extract(self, wave_frame):
    padded_frame = np.pad(wave_frame, (4, 3), mode='constant', constant_values=0)
    fft_features = []
    zcr_features = []
    slope_features = []
    dom_freq_features = []
    spectral_entropy_features = []
    stride_window = int(self.frame_size * self.stride_size)
    window_size = len(padded_frame) - (self.window_count * stride_window)
    for i in range(self.window_count + 1):
        start = i * stride_window
        end = start + window_size
        if end > len(padded_frame):
            break
        window = padded_frame[start:end]
        _, mirrored_window = self._transform_symmetric_mirror(window)
        num_bins = self.fft_size // 2 + 1
        fft_result = np.abs(np.fft.fft(mirrored_window, n=self.fft_size))[:num_bins]
        fft_features.append(fft_result)
    fft_features = np.array(fft_features)
    fft_features = fft_features[:-1, 1:]
    bin_count_pool = 16
    pool_size = self.fft_size // (2 * bin_count_pool)
    fft_features = fft_features.reshape(self.window_count, bin_count_pool, pool_size).mean(axis=-1)
    _, kurt_features = self._transform_kurtosis(wave_frame)
    for i in range(self.window_count):
        start = i * stride_window
        end = start + window_size
        if end > len(padded_frame):
            break
        window = padded_frame[start:end]
        _, zcr = self._transform_zero_crossing_rate(window)
        zcr_features.append(zcr)
        _, slope = self._transform_slope_changes(window)
        slope_features.append(slope)
        _, dom_freqs = self._transform_dominant_frequency(window)
        dom_freq_features.append(dom_freqs)
        _, spectral_entropy_val = self._transform_spectral_entropy(window)
        spectral_entropy_features.append(spectral_entropy_val)
    zcr_features = np.array(zcr_features).reshape(-1, 1)
    slope_features = np.array(slope_features).reshape(-1, 1)
    dom_freq_features = np.array(dom_freq_features)
    spectral_entropy_features = np.array(spectral_entropy_features).reshape(-1, 1)
    result_wave = np.hstack([fft_features, kurt_features, zcr_features, slope_features, dom_freq_features, spectral_entropy_features])
    return wave_frame, result_wave

def _transform_pir_feature_extract_fixed_point(self, wave_frame):
    wave_frame = wave_frame - 128    # Remove DC Offset to centre signal around zero.
    padded_frame = np.pad(wave_frame, (4, 3), mode='constant', constant_values=0)

    fft_features = []
    zcr_features = []
    slope_features = []
    dom_freq_features = []        
    stride_window = int(self.frame_size * self.stride_size)
    window_size = len(padded_frame) - (self.window_count*stride_window) 
    for i in range(self.window_count + 1):            
        start = i * stride_window
        end = start + window_size
        if end > len(padded_frame):  # Ensure window does not exceed length
            break
        window = padded_frame[start:end]  # Extract window
        window = window * 128
        # Symmetrically mirror the window to 128 samples
        _, mirrored_window = self._transform_symmetric_mirror(window)
        num_bins = self.fft_size// 2 + 1 
        # Perform FFT and take absolute values            
        inst = dsp.arm_rfft_instance_q15()
        dsp.arm_rfft_init_q15(inst, self.fft_size,0,1)         # forward FFT
        fft_res = dsp.arm_rfft_q15(inst, mirrored_window)
        mag_res = dsp.arm_cmplx_mag_q15(fft_res)[:num_bins]
        mag_res = mag_res * 128       # Scale the magnitude by 128
        fft_features.append(mag_res)  # Store FFT output            

    fft_features = np.array(fft_features)        
    fft_features = fft_features[:-1,1:]  # Remove the last one to maintain window_count
    bin_count_pool = 16
    pool_size      = self.fft_size//(2*bin_count_pool)
    fft_features = np.clip(
        np.round(fft_features.reshape(self.window_count, bin_count_pool, pool_size).mean(axis=-1)), 
        np.iinfo(np.int16).min, 
        np.iinfo(np.int16).max
        ).astype(np.int16)  # Average Pooling                
    for i in range(self.window_count):
        start = i * stride_window
        end = start + window_size
        if end > len(padded_frame):  # Ensure window does not exceed length
            break
        window = padded_frame[start:end]        
        _,zcr = self._transform_zero_crossing_rate_fixed_point(window)
        zcr_features.append(zcr)
        _,slope = self._transform_slope_changes_fixed_point(window)
        slope_features.append(slope)
        _,dom_freqs = self._transform_dominant_frequency_fixed_point(window)
        dom_freq_features.append(dom_freqs)
    zcr_features = np.array(zcr_features)
    slope_features = np.array(slope_features)
    dom_freq_features = np.array(dom_freq_features)
    zcr_features = zcr_features.reshape(-1, 1)
    slope_features = slope_features.reshape(-1, 1)        
    # Feature Extraction for PIR is still in development phase. Support for fixed point kurtosis and spectral entropy will be added later.
    kurt_features = np.zeros((25, 4), dtype=int)
    spectral_entropy_features = np.zeros((25, 1), dtype=int)
    result_wave = np.hstack([fft_features, kurt_features, zcr_features, slope_features, dom_freq_features, spectral_entropy_features])    
    return wave_frame, result_wave

def _transform_shape(self, concatenated_features, raw_frames):
    """Rearrange the shape from (C,N,W) to (N,C,W,H)."""
    N = concatenated_features.shape[1]
    if concatenated_features.ndim == 3:
        x_temp = concatenated_features.transpose(1, 0, 2).reshape(N, self.ch, self.wl, self.hl)
    else:
        x_temp = concatenated_features.transpose(1, 0, 2, 3)  # PIR Detection gives concatenated_features as (N,C,W,H)
    x_temp_raw_out = raw_frames.transpose(1, 0, 2)
    return x_temp, x_temp_raw_out