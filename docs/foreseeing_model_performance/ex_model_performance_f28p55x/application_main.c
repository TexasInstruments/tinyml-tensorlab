//#############################################################################
//
// FILE:   application_main.c
//
//!
//! This example shows the use of feature_extract library files in to process
//! motor_fault_classification data set. The example is developed on f28p55
//! launchpad. Please set the "TMS320F28P550SJ9_LaunchPad.ccxml" in
//! "targetConfigs" as the active configuration.
//
//#############################################################################
//
//
// $Copyright:  $
//#############################################################################

#include "driverlib.h"
#include "device.h"
#include "board.h"
#include "c2000ware_libraries.h"
#include "user_input_config.h"
#include "tvmgen_default.h"
#include <stdio.h>
#include "profiling.h"

#ifdef TVMGEN_DEFAULT_TI_NPU
extern void TI_NPU_init();
extern volatile int32_t tvmgen_default_finished;
#endif
uint32_t t0, t1;

void boards_devices_init(){
    // Initialize device clock and peripherals
    Device_init();

    // Disable pin locks and enable internal pull-ups.
    Device_initGPIO();

    // Initialize PIE and clear PIE registers. Disables CPU interrupts.
    Interrupt_initModule();

    // Initialize the PIE vector table with pointers to the shell Interrupt Service Routines (ISR).
    Interrupt_initVectorTable();

    // PinMux and Peripheral Initialization
    Board_init();

    // C2000Ware Library initialization
    C2000Ware_libraries_init();


    Interrupt_enableGlobal(); // Enable CPU interrupt

    // Enable Global Interrupt (INTM) and real time interrupt (DBGM)
    EINT;
    ERTM;
    return;
}

// Prepare array for input and output of model
int8_t input[FE_STACKING_CHANNELS][FE_HL][FE_STACKING_FRAME_WIDTH];
int8_t output[FE_NN_OUT_SIZE];

void main(void)
{
    boards_devices_init();
    
    // Provide (void*) pointers of input and output to TI - NPU
    struct tvmgen_default_inputs inputs = { &input};
    struct tvmgen_default_outputs outputs = { &output };

    // Initialize TI-NPU
    #ifdef TVMGEN_DEFAULT_TI_NPU
    TI_NPU_init();
    #endif
    TEUtils_cycle_init();
    // Read the cycles at start of model inferencing
    t0 = TEUtils_cycle_read();
    // Run the model on TI-NPU
    tvmgen_default_run(&inputs, &outputs);
    // Wait for the model to finish
    #ifdef TVMGEN_DEFAULT_TI_NPU
    while (!tvmgen_default_finished);
    #endif
    // Read the cycles at end of model inferencing
    t1 = TEUtils_cycle_read();

    printf("Cycles: %ld\n", TEUtils_cycle_diff(t0, t1));

    return;
}

//
// End of File
//

