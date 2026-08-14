//#############################################################################
//
// FILE:   device.h
//
// TITLE:  Device setup for examples.
//
//#############################################################################
//
//
// //
//	Copyright: Copyright (C) Texas Instruments Incorporated
//	All rights reserved not granted herein.
//
//  Redistribution and use in source and binary forms, with or without 
//  modification, are permitted provided that the following conditions 
//  are met:
//
//  Redistributions of source code must retain the above copyright 
//  notice, this list of conditions and the following disclaimer.
//
//  Redistributions in binary form must reproduce the above copyright
//  notice, this list of conditions and the following disclaimer in the 
//  documentation and/or other materials provided with the   
//  distribution.
//
//  Neither the name of Texas Instruments Incorporated nor the names of
//  its contributors may be used to endorse or promote products derived
//  from this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
//  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

//#############################################################################

#ifndef __DEVICE_H__
#define __DEVICE_H__

#ifdef __cplusplus
extern "C"
{
#endif

//
// Included Files
//
#include "driverlib.h"
#include <cpy_tbl.h>
#include <string.h>

//*****************************************************************************
//
// Check for invalid compile flags
//
//*****************************************************************************
#if ((!defined(__CPU3__)) && defined(__C29_OPTF64__))
#error "Invalid FPU Configuration for the CPU"
#endif

//*****************************************************************************
//
// Defines related to clock configuration
//
//*****************************************************************************

#define DEVICE_SYSCLK_FREQ      200000000

//
// To use XTAL as the clock source, uncomment #define USE_PLL_SRC_XTAL,
// and comment the #define USE_PLL_SRC_INTOSC
//
//#define USE_PLL_SRC_XTAL
#define USE_PLL_SRC_INTOSC

#if defined(USE_PLL_SRC_INTOSC)
#define DEVICE_OSCSRC_FREQ		10000000
#elif defined(USE_PLL_SRC_XTAL)
#define DEVICE_OSCSRC_FREQ		25000000
#endif

//*****************************************************************************
//
// Macro to call SysCtl_delay() to achieve a delay in microseconds. The macro
// will convert the desired delay in microseconds to the count value expected
// by the function. \b x is the number of microseconds to delay.
//
//*****************************************************************************
#define DEVICE_DELAY_US(x) SysCtl_delay(((((long double)(x)) / (1000000.0L /  \
                              (long double)DEVICE_SYSCLK_FREQ)) - 11.0L) / 4.0L)

//*****************************************************************************
//
// Defines for pin numbers
//
//*****************************************************************************

#ifdef _LAUNCHXL_F29H85X

//
// LaunchPad
//

//
// LEDs
//
#define DEVICE_GPIO_PIN_LED1        19U             // GPIO number for LED4
#define DEVICE_GPIO_PIN_LED2        62U             // GPIO number for LED5
#define DEVICE_GPIO_CFG_LED1        GPIO_19_GPIO19  // "pinConfig" for LED4
#define DEVICE_GPIO_CFG_LED2        GPIO_62_GPIO62  // "pinConfig" for LED5

#else

//
// ControlSOM
//

//
// LEDs
//
#define DEVICE_GPIO_PIN_LED1        23U             // GPIO number for LED3
#define DEVICE_GPIO_PIN_LED2        9U              // GPIO number for LED4
#define DEVICE_GPIO_CFG_LED1        GPIO_23_GPIO23  // "pinConfig" for LED3
#define DEVICE_GPIO_CFG_LED2        GPIO_9_GPIO9    // "pinConfig" for LED4

#endif

//
// UART 
//
#define DEVICE_GPIO_PIN_UARTA_TX  42U               // GPIO number for UARTA TX
#define DEVICE_GPIO_PIN_UARTA_RX  43U               // GPIO number for UARTA RX
#define DEVICE_GPIO_CFG_UARTA_TX  GPIO_42_UARTA_TX  // "pinConfig" for UARTA TX
#define DEVICE_GPIO_CFG_UARTA_RX  GPIO_43_UARTA_RX  // "pinConfig" for UARTA RX

//*****************************************************************************
//
// Defines related to Flash Support
//
//*****************************************************************************
#define DEVICE_FLASH_WAITSTATES  3

//*****************************************************************************
//
// An equivalent function of copy_in() to be used in multi-core applications.
// It takes the address of a linker-generated copy table as an argument and
// processes the copy table data object to perform the copy of each object
// component specified in the copy table.
//
//*****************************************************************************
void Device_copy_in(COPY_TABLE *tp);

//*****************************************************************************
//
// Function Prototypes
//
//*****************************************************************************
//*****************************************************************************
//
//! \addtogroup device_api
//! @{
//
//*****************************************************************************

//*****************************************************************************
//
//! @brief Function to initialize the device. Primarily initializes system
//!  control to a known state by disabling the watchdog, setting up the
//!  SYSCLKOUT frequency, and enabling the clocks to the peripherals.
//!
//! \param None.
//! \return None.
//
//*****************************************************************************
extern void Device_init(void);

//*****************************************************************************
//!
//!
//! @brief Function to turn on all peripherals, enabling reads and writes to the
//! peripherals' registers.
//!
//! Note that to reduce power, unused peripherals should be disabled.
//!
//! @param None
//! @return None
//
//*****************************************************************************
extern void Device_enableAllPeripherals(void);

//*****************************************************************************
//!
//!
//! @brief Function to disable pin locks on GPIOs.
//!
//! @param None
//! @return None
//
//*****************************************************************************
extern void Device_initGPIO(void);

//*****************************************************************************
//!
//! @brief Function to verify the XTAL frequency
//!
//! \param freq is the XTAL frequency in MHz
//! \return The function returns true if the the actual XTAL frequency matches
//! with the input value.
//
//*****************************************************************************
extern bool Device_verifyXTAL(float freq);

//*****************************************************************************
//!
//!
//! @brief Function to boot CPU2.
//!
//! @param copyTable is the copy table created by the CPU2 linker.
//! @param reset_vector is the address to which the CPU2 should boot.
//! Loaded by CPU1.LINK2 application code prior to releasing this CPU's reset.
//! @param link is the SSU link to which the CPU2 should boot.
//! Loaded by CPU1.LINK2 application code prior to releasing this CPU's reset.
//!
//! This function must be called after Device_init function.
//!
//! @return None
//
//*****************************************************************************
extern void
Device_bootCPU2(COPY_TABLE *copyTable, uint32_t reset_vector, SSU_Link link);

//*****************************************************************************
//!
//!
//! @brief Function to boot CPU3.
//!
//! @param copyTable is the copy table created by the CPU3 linker. When the
//! FLASH bank Mode 2/3 is used, CPU3 has dedicated FLASH region where its
//! application code is loaded. In such a case, pass the value '0' to the
//! \e copyTable parameter.
//! @param reset_vector is the address to which the CPU3 should boot.
//! Loaded by CPU1.LINK2 application code prior to releasing this CPU's reset.
//! @param link is the SSU link to which the CPU3 should boot.
//! Loaded by CPU1.LINK2 application code prior to releasing this CPU's reset.
//!
//! This function must be called after Device_init function.
//!
//! @return None
//
//*****************************************************************************
extern void
Device_bootCPU3(COPY_TABLE *copyTable, uint32_t reset_vector, SSU_Link link);

//*****************************************************************************
//!
//!
//! @brief Errata Workaround for System: Pending Misaligned Reads in the
//! Pipeline After CPU Goes to Fault State Preventing NMI Vector Fetch.
//!
//! @param None
//!
//! This function is a workaround for the issue mentioned in the Errata under
//! the title "System: Pending Misaligned Reads in the Pipeline After CPU Goes
//! to Fault State Preventing NMI Vector Fetch". It is called by default in the
//! Device_init function. This is necessary for the NMI handler to execute as
//! expected in case of three or more back-to-back C29 CPU faults caused by
//! misaligned reads.
//!
//! @return None
//
//*****************************************************************************
extern void
Device_errataWorkaroundNMIVectorFetch(void);

//*****************************************************************************
//!
//! @brief Error handling function to be called when an ASSERT is violated
//!
//! @param *filename File name in which the error has occurred
//! @param line Line number within the file
//! @return None
//
//*****************************************************************************
extern void __error__(const char *filename, uint32_t line);

//*****************************************************************************
//
// Close the Doxygen group.
//! @}
//
//*****************************************************************************

#ifdef __cplusplus
}
#endif

#endif // __DEVICE_H__
