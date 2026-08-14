/*
 *  Created on: Nov 20, 2025
 */
// #############################################################################
//
//  FILE:   feature_extract_c28.c
//
//!
//! This contains definition of CFFT API functions required to feature
//! extraction operations for c28x device
//!
//
// #############################################################################
//
// Copyright (C) 2025 Texas Instruments Incorporated - http://www.ti.com
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//   Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
//   Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the
//   distribution.
//
//   Neither the name of Texas Instruments Incorporated nor the names of
//   its contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// $
// #############################################################################

#include "fpu32/fpu_cfft.h"
#include "fpu32/C28x_FPU_FastRTS.h"

/**
 * @brief Performs Complex Fast Fourier Transform (CFFT) on input data
 * 
 * @param fft_input_buffer Input buffer containing time-domain data
 * @param fft_output_buffer Output buffer to store frequency-domain results
 * @param frame_size Size of the input frame (must be power of 2)
 * @param fft_stages Number of FFT stages (log2 of frame size)
 * 
 * This function:
 * 1. Configures the CFFT object with input parameters
 * 2. Performs complex FFT transformation
 * 3. Unpacks the results
 * 4. Calculates magnitude spectrum
 * 5. Handles special case for odd number of FFT stages
 */
void FE_cfft(float *fft_input_buffer, float *fft_output_buffer, uint16_t frame_size, uint16_t fft_stages)
{

    CFFT_F32_STRUCT cfft;
    CFFT_F32_STRUCT_Handle cfft_handle;
    
    cfft_handle = &cfft;
    // Configure the FFT object
    CFFT_f32_setInputPtr(cfft_handle, fft_input_buffer);
    CFFT_f32_setOutputPtr(cfft_handle, fft_output_buffer);
    CFFT_f32_setTwiddlesPtr(cfft_handle, CFFT_f32_twiddleFactors);
    CFFT_f32_setStages(cfft_handle, (fft_stages - 1U));
    CFFT_f32_setFFTSize(cfft_handle, (frame_size >> 1));

    CFFT_f32t(cfft_handle);
    CFFT_f32_unpack(cfft_handle);

    float *p_temp = CFFT_f32_getCurrInputPtr(cfft_handle);
    CFFT_f32_setCurrInputPtr(cfft_handle, CFFT_f32_getCurrOutputPtr(cfft_handle));
    CFFT_f32_setCurrOutputPtr(cfft_handle, p_temp);

#ifdef __TMS320C28XX_TMU__
    CFFT_f32_mag_TMU0(cfft_handle);
#else
    CFFT_f32_mag(cfft_handle);
#endif
    float last_element = fft_output_buffer[frame_size];
    if(fft_stages%2!=0){
        memcpy(fft_input_buffer, fft_output_buffer, frame_size * sizeof(float));
        last_element = fft_input_buffer[frame_size];
    }
    last_element = last_element < 0 ? -last_element : last_element;
    fft_input_buffer[frame_size/2] = last_element;
    return;
}
