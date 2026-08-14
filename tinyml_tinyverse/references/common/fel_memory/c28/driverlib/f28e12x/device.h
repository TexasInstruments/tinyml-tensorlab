//#############################################################################
//
// FILE:   device.h
//
// TITLE:  Device setup for examples.
//
//#############################################################################
//
//
// $Copyright:
// Copyright (C) 2026 Texas Instruments Incorporated - http://www.ti.com/
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

//
// Check for invalid compile flags
//
#if defined(__TMS320C28XX_FPU64__)
#error "Invalid FPU Configuration"
#endif

//*****************************************************************************
//
// Defines related to Device package information
//
//*****************************************************************************

//
// The Unbonded GPIO pull ups is different for different package type. If the
// package type is different from the default package tyep (48PT), then define 
// the DEV_PKG_TYPE macro as a predefined symbol in the projectspec.
// (Add "--define=DEV_PKG_TYPE=value(based on the package type defined below)" 
// in the compiler flags).

// 48PT
// Unbonded GPIOs: 9, 10, 11, 23, 30, 39, 40, 41, 243
//
#define DEV_PKG_48PT 0

// 32RHB
// Unbonded GPIOs: 2, 6, 9, 10, 12, 13, 16, 23, 30, 33, 39, 40, 41, 45, 243
//
#define DEV_PKG_32RHB 1

// 32VFC
// Unbonded GPIOs: 2, 6, 7, 9, 10, 12, 13, 16, 23, 30, 33, 39, 40, 41, 43, 45, 
//                 243
//
#define DEV_PKG_32VFC 2

#ifndef DEV_PKG_TYPE
#define DEV_PKG_TYPE DEV_PKG_48PT
#endif

//*****************************************************************************
//
// Defines for pin numbers
//
//*****************************************************************************

//
// LaunchPad
//

//
// LEDs
//
#define DEVICE_GPIO_PIN_LED1        45U                 // GPIO number for LED4
#define DEVICE_GPIO_PIN_LED2        33U                 // GPIO number for LED5
#define DEVICE_GPIO_CFG_LED1        GPIO_45_GPIO45      // "pinConfig" for LED4
#define DEVICE_GPIO_CFG_LED2        GPIO_33_GPIO33      // "pinConfig" for LED5

//
// SCI for USB-to-UART adapter on FTDI chip
//
#define DEVICE_GPIO_PIN_SCIRXDA     35U                 // GPIO number for SCIA RX
#define DEVICE_GPIO_PIN_SCITXDA     37U                 // GPIO number for SCIA TX
#define DEVICE_GPIO_CFG_SCIRXDA     GPIO_35_SCIA_RX     // "pinConfig" for SCIA RX
#define DEVICE_GPIO_CFG_SCITXDA     GPIO_37_SCIA_TX     // "pinConfig" for SCIA TX

//
// I2CA
//
#define DEVICE_GPIO_PIN_SDAA        19U                // GPIO number for I2CA SDA
#define DEVICE_GPIO_PIN_SCLA        18U                // GPIO number for I2CA SCL
#define DEVICE_GPIO_CFG_SDAA        GPIO_19_I2CA_SDA   // "pinConfig" for I2CA SDA
#define DEVICE_GPIO_CFG_SCLA        GPIO_18_I2CA_SCL   // "pinConfig" for I2CA SCL

//
// eQEP1
//
#define DEVICE_GPIO_PIN_EQEP1A      6U  // GPIO number for EQEP 1A
#define DEVICE_GPIO_PIN_EQEP1B      7U  // GPIO number for EQEP 1B
#define DEVICE_GPIO_PIN_EQEP1I      43U  // GPIO number for EQEP 1I
#define DEVICE_GPIO_CFG_EQEP1A      GPIO_6_EQEP1_A  // "pinConfig" for EQEP 1A
#define DEVICE_GPIO_CFG_EQEP1B      GPIO_7_EQEP1_B  // "pinConfig" for EQEP 1B
#define DEVICE_GPIO_CFG_EQEP1I      GPIO_43_EQEP1_INDEX  // "pinConfig" for EQEP 1I

//*****************************************************************************
//
// Defines related to clock configuration
//
//*****************************************************************************

//
// To use SYSOSCDIV4 as the clock source, comment the #define USE_PLL_SRC_XTAL,
// and uncomment the #define USE_PLL_SRC_SYSOSCDIV4
//
#define USE_PLL_SRC_XTAL
//#define USE_PLL_SRC_SYSOSCDIV4


#if defined(USE_PLL_SRC_XTAL)
//
// 20MHz XTAL is used as the PLL source. The value can go upto
// 50MHz 
// For use with SysCtl_getClock().
//
#define DEVICE_OSCSRC_FREQ          20000000U

//
// Define to pass to SysCtl_setClock(). Will configure the clock as follows:
// PLLSYSCLK = 20MHz (XTAL_OSC) * 32 (QDIV) / (2 (PDIV) * 2 (RDIVCLK0) * 1(SYSDIV))
//
// 160MHz SYSCLK frequency based on the above DEVICE_SETCLOCK_CFG. Update the
// code below if a different clock configuration is used!
//
#define DEVICE_SYSCLK_FREQ          ((DEVICE_OSCSRC_FREQ * 32) / (2 * 2 * 1))

#elif defined(USE_PLL_SRC_SYSOSCDIV4)
//
// 8MHz SYSOSCDIV4 on the device is used as the PLL source.
// For use with SysCtl_getClock().
//
#define DEVICE_OSCSRC_FREQ          8000000U

//
// Define to pass to SysCtl_setClock(). Will configure the clock as follows:
// PLLSYSCLK = 8MHz (SYSOSCDIV4) * 40 (QDIV) / (1 (PDIV) * 2 (RDIVCLK0) * 1(SYSDIV))
//
// 160MHz SYSCLK frequency based on the above DEVICE_SETCLOCK_CFG. Update the
// code below if a different clock configuration is used!
//
#define DEVICE_SYSCLK_FREQ          ((DEVICE_OSCSRC_FREQ * 40) / (1 * 2 * 1))
#endif

//
// 40MHz LSPCLK frequency based on the above DEVICE_SYSCLK_FREQ and a default
// low speed peripheral clock divider of 4. Update the code below if a
// different LSPCLK divider is used!
//
#define DEVICE_LSPCLK_FREQ          (DEVICE_SYSCLK_FREQ / 4)

//*****************************************************************************
//
// Macro to call SysCtl_delay() to achieve a delay in microseconds. The macro
// will convert the desired delay in microseconds to the count value expected
// by the function. \b x is the number of microseconds to delay.
//
//*****************************************************************************
#define DEVICE_DELAY_US(x) SysCtl_delay(((((long double)(x)) / (1000000.0L /  \
                              (long double)DEVICE_SYSCLK_FREQ)) - 9.0L) / 5.0L)

//*****************************************************************************
//
// Defines, Globals, and Header Includes related to Flash Support
//
//*****************************************************************************
#ifdef _FLASH
#include <stddef.h>

#ifndef CMDTOOL
extern uint16_t RamfuncsLoadStart;
extern uint16_t RamfuncsLoadEnd;
extern uint16_t RamfuncsLoadSize;
extern uint16_t RamfuncsRunStart;
extern uint16_t RamfuncsRunEnd;
extern uint16_t RamfuncsRunSize;
#endif

#endif

#if ((DEVICE_SYSCLK_FREQ / 1000000) <= 160U && (DEVICE_SYSCLK_FREQ / 1000000) > 120U)
    #define DEVICE_FLASH_WAITSTATES 3
#elif (((DEVICE_SYSCLK_FREQ / 1000000) <= 120U) && ((DEVICE_SYSCLK_FREQ / 1000000) >= 80U))
    #define DEVICE_FLASH_WAITSTATES 2
#endif

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
//!  control to aknown state by disabling the watchdog, setting up the
//!  SYSCLKOUT frequency, and enabling the clocks to the peripherals.
//!
//! \param None.
//! \return None.
//
//*****************************************************************************
extern void Device_init(void);

//*****************************************************************************
//
//!
//! @brief Function to verify the XTAL frequency
//! \param freq is the XTAL frequency in MHz
//! \return The function return true if the the actual XTAL frequency matches with the
//! input value
//
//*****************************************************************************
extern bool Device_verifyXTAL(float freq);

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
//!
//! @brief Function to enable pullups for the unbonded GPIOs.
//!
//! @param packageType is the current package type of the device
//! @return None
//
//*****************************************************************************
extern void Device_enableUnbondedGPIOPullups(uint16_t packageType);

//*****************************************************************************
//!
//!
//! @brief Function to update peripheral configuraiton register.
//!
//! @param packageType is the current package type of the device
//! @return None
//
//*****************************************************************************
extern void Device_updatePeripheralConfig(uint16_t packageType);

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
