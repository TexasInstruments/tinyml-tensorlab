/*
 *  Copyright (C) 2026 Texas Instruments Incorporated
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>

/* ARM CMSIS-DSP Header files */
#include "arm_const_structs.h"
#include "arm_math.h"

void FE_cfft(float *input, float *output, uint16_t frame_size, uint16_t fe_stages)
{
    for (uint16_t i = 0; i < frame_size; i++)
    {
        output[2 * i]     = input[i];
        output[2 * i + 1] = 0.0f;
    }

    const arm_cfft_instance_f32 *cfft_instance;

    switch (frame_size)
    {
        case 32:
            cfft_instance = &arm_cfft_sR_f32_len32;
            break;
        case 64:
            cfft_instance = &arm_cfft_sR_f32_len64;
            break;
        case 128:
            cfft_instance = &arm_cfft_sR_f32_len128;
            break;
        case 256:
            cfft_instance = &arm_cfft_sR_f32_len256;
            break;
        case 512:
            cfft_instance = &arm_cfft_sR_f32_len512;
            break;
        case 1024:
            cfft_instance = &arm_cfft_sR_f32_len1024;
            break;
        case 2048:
            cfft_instance = &arm_cfft_sR_f32_len2048;
            break;
        case 4096:
            cfft_instance = &arm_cfft_sR_f32_len4096;
            break;
        default:
            return;
    }
    arm_cfft_f32(cfft_instance, output, 0, 1);
    arm_cmplx_mag_f32(output, input, frame_size);
}
