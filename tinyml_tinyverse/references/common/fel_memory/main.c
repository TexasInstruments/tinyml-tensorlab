/*
 *  Copyright (C) 2025 Texas Instruments Incorporated
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

/* SDK includes */
#include "stdio.h"

/* Example includes */
#include "feature_extract.h"

/* ========================================================================== */
/*                 Private Definitions and Macros                             */
/* ========================================================================== */

/* ========================================================================== */
/*                 Private Typedefs                                           */
/* ========================================================================== */

/* ========================================================================== */
/*                 Private Function Prototype                                 */
/* ========================================================================== */

/* ========================================================================== */
/*                 Private Variable Declaration                               */
/* ========================================================================== */

feature_extraction App_Ai_FeHandle;

float                   raw_input_test[];
model_input_t           model_test_input[];

float              App_Ai_ScratchBuffer[FE_FRAME_SIZE * 4];
model_input_t      App_Ai_ModelInput[1 * FE_STACKING_CHANNELS * FE_STACKING_FRAME_WIDTH * 1];

/* ========================================================================== */
/*                 Public Functions                                           */
/* ========================================================================== */

int main()
{
    
    int error       = 0;

    feature_extraction_handle feHandlePtr = &App_Ai_FeHandle;
    feHandlePtr->scratch_buffer = &App_Ai_ScratchBuffer[0];
    
    FE_allocFeatureExtract(feHandlePtr);
    
    feHandlePtr->input_buffer            = &raw_input_test[0];
    feHandlePtr->output_buffer           = &App_Ai_ModelInput[0];
    feHandlePtr->history_buffer          = &model_test_input[0];
    feHandlePtr->test_feature_extraction = false;
    

    FE_runFeatureExtract(feHandlePtr);
    
    error = FE_compareModelInput(model_test_input, App_Ai_ModelInput);
    printf("Error: %d\n", error);
        
    return 0;
}

/* ========================================================================== */
/*                 Private Functions                                          */
/* ========================================================================== */

/* ========================================================================== */
/*                 Private Callback Handlers                                  */
/* ========================================================================== */

/* ========================================================================== */
/*                 Test/Debug/Other Sections                                  */
/* ========================================================================== */
