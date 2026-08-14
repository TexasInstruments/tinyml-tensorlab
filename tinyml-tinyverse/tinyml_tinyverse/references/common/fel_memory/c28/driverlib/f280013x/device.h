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
#if defined(__TMS320C28XX_TMU1__)
#error "Invalid TMU Configuration"
#endif
#if defined(__TMS320C28XX_IDIV__)
#error "Invalid IDIV Configuration"
#endif

//*****************************************************************************
//
// Defines for pin numbers
//
//*****************************************************************************
#ifdef _LAUNCHXL_F2800137
//
// LaunchPad
//
//
// LEDs
//
#define DEVICE_GPIO_PIN_LED1        20U             // GPIO number for LED4
#define DEVICE_GPIO_PIN_LED2        22U             // GPIO number for LED5
#define DEVICE_GPIO_CFG_LED1        GPIO_20_GPIO20  // "pinConfig" for LED4
#define DEVICE_GPIO_CFG_LED2        GPIO_22_GPIO22  // "pinConfig" for LED5

//
// CANA
//
#define DEVICE_GPIO_PIN_CANRXA      5U              // GPIO number for CANA RX
#define DEVICE_GPIO_PIN_CANTXA      4U              // GPIO number for CANA TX
#define DEVICE_GPIO_CFG_CANRXA      GPIO_5_CANA_RX  // "pinConfig" for CANA RX
#define DEVICE_GPIO_CFG_CANTXA      GPIO_4_CANA_TX  // "pinConfig" for CANA TX

//
// SCI for USB-to-UART adapter on XDS110 chip
//
#define DEVICE_GPIO_PIN_SCIRXDA     28U             // GPIO number for SCIA RX
#define DEVICE_GPIO_PIN_SCITXDA     29U             // GPIO number for SCIA TX
#define DEVICE_GPIO_CFG_SCIRXDA     GPIO_28_SCIA_RX // "pinConfig" for SCIA RX
#define DEVICE_GPIO_CFG_SCITXDA     GPIO_29_SCIA_TX // "pinConfig" for SCIA TX

//
// I2CA
//
#define DEVICE_GPIO_PIN_SDAA        32U  // GPIO number for I2C SDAA
#define DEVICE_GPIO_PIN_SCLA        33U  // GPIO number for I2C SCLA
#define DEVICE_GPIO_CFG_SDAA        GPIO_32_I2CA_SDA  // "pinConfig" for I2C SDAA
#define DEVICE_GPIO_CFG_SCLA        GPIO_33_I2CA_SCL  // "pinConfig" for I2C SCLA

#define DEVICE_GPIO_PIN_SDAA_2      19U  // GPIO number for I2C SDAA
#define DEVICE_GPIO_PIN_SCLA_2      18U  // GPIO number for I2C SCLA
#define DEVICE_GPIO_CFG_SDAA_2      GPIO_19_I2CA_SDA  // "pinConfig" for I2C SDAA
#define DEVICE_GPIO_CFG_SCLA_2      GPIO_18_I2CA_SCL  // "pinConfig" for I2C SCLA

#define DEVICE_GPIO_PIN_SDAB        28U  // GPIO number for I2C SDAA
#define DEVICE_GPIO_PIN_SCLB        29U  // GPIO number for I2C SCLA
#define DEVICE_GPIO_CFG_SDAB        GPIO_28_I2CB_SDA  // "pinConfig" for I2C SDAA
#define DEVICE_GPIO_CFG_SCLB        GPIO_29_I2CB_SCL  // "pinConfig" for I2C SCLA

//
// SPIA
//
#define DEVICE_GPIO_PIN_SPICLKA     9U   // GPIO number for SPI CLKA
#define DEVICE_GPIO_PIN_SPISIMOA    8U   // GPIO number for SPI SIMOA
#define DEVICE_GPIO_PIN_SPISOMIA    17U  // GPIO number for SPI SOMIA
#define DEVICE_GPIO_PIN_SPISTEA     5U   // GPIO number for SPI STEA
#define DEVICE_GPIO_CFG_SPICLKA     GPIO_9_SPIA_CLK    // "pinConfig" for SPI CLKA
#define DEVICE_GPIO_CFG_SPISIMOA    GPIO_8_SPIA_SIMO   // "pinConfig" for SPI SIMOA
#define DEVICE_GPIO_CFG_SPISOMIA    GPIO_17_SPIA_SOMI  // "pinConfig" for SPI SOMIA
#define DEVICE_GPIO_CFG_SPISTEA     GPIO_5_SPIA_STE    // "pinConfig" for SPI STEA

//
// eQEP1
//
#define DEVICE_GPIO_PIN_EQEP1A      40U  // GPIO number for EQEP 1A
#define DEVICE_GPIO_PIN_EQEP1B      41U  // GPIO number for EQEP 1B
#define DEVICE_GPIO_PIN_EQEP1I      39U  // GPIO number for EQEP 1I
#define DEVICE_GPIO_CFG_EQEP1A      GPIO_40_EQEP1_A  // "pinConfig" for EQEP 1A
#define DEVICE_GPIO_CFG_EQEP1B      GPIO_41_EQEP1_B  // "pinConfig" for EQEP 1B
#define DEVICE_GPIO_CFG_EQEP1I      GPIO_39_EQEP1_INDEX  // "pinConfig" for EQEP 1I

#else
//
// ControlCARD
//

//
// LEDs
//
#define DEVICE_GPIO_PIN_LED1        24U             // GPIO number for LED1
#define DEVICE_GPIO_PIN_LED2        39U             // GPIO number for LED2
#define DEVICE_GPIO_CFG_LED1        GPIO_24_GPIO24  // "pinConfig" for LED1
#define DEVICE_GPIO_CFG_LED2        GPIO_39_GPIO39  // "pinConfig" for LED2

//
// CANA
//
#define DEVICE_GPIO_PIN_CANTXA      32U              // GPIO number for CANTXA
#define DEVICE_GPIO_PIN_CANRXA      33U              // GPIO number for CANRXA
#define DEVICE_GPIO_CFG_CANRXA      GPIO_33_CANA_RX  // "pinConfig" for CANA RX
#define DEVICE_GPIO_CFG_CANTXA      GPIO_32_CANA_TX  // "pinConfig" for CANA TX

//
// SCI for USB-to-UART adapter on FTDI chip
//
#define DEVICE_GPIO_PIN_SCIRXDA     28U             // GPIO number for SCI RX
#define DEVICE_GPIO_PIN_SCITXDA     29U             // GPIO number for SCI TX
#define DEVICE_GPIO_CFG_SCIRXDA     GPIO_28_SCIA_RX // "pinConfig" for SCI RX
#define DEVICE_GPIO_CFG_SCITXDA     GPIO_29_SCIA_TX // "pinConfig" for SCI TX

//
// I2C
//
#define DEVICE_GPIO_PIN_SDAA        32U  // GPIO number for I2C SDAA
#define DEVICE_GPIO_PIN_SCLA        33U  // GPIO number for I2C SCLA
#define DEVICE_GPIO_CFG_SDAA        GPIO_32_I2CA_SDA  // "pinConfig" for I2C SDAA
#define DEVICE_GPIO_CFG_SCLA        GPIO_33_I2CA_SCL  // "pinConfig" for I2C SCLA

//I2CB GPIO pins
#define DEVICE_GPIO_PIN_SDAB    2
#define DEVICE_GPIO_PIN_SCLB    3

#define DEVICE_GPIO_CFG_SDAB GPIO_2_I2CB_SDA
#define DEVICE_GPIO_CFG_SCLB GPIO_3_I2CB_SCL

#endif

//*****************************************************************************
//
// Defines related to clock configuration
//
//*****************************************************************************

//
// To use XTAL as the clock source, uncomment #define USE_PLL_SRC_XTAL,
// and comment the #define USE_PLL_SRC_INTOSC
//
//#define USE_PLL_SRC_XTAL
#define USE_PLL_SRC_INTOSC

//
// To use CPU frequency as 100MHZ, uncomment #define CPU_FRQ_100MHZ,
// and comment the #define CPU_FRQ_120MHZ
//
//#define CPU_FRQ_100MHZ
#define CPU_FRQ_120MHZ

#if defined(USE_PLL_SRC_XTAL)
//
// 20MHz XTAL on controlCARD is used as the PLL source.
// For use with SysCtl_getClock().
//
#define DEVICE_OSCSRC_FREQ          20000000U


#if defined(CPU_FRQ_100MHZ)

//
// Define to pass to SysCtl_setClock(). Will configure the clock as follows:
// PLLSYSCLK = 20MHz (XTAL_OSC) * 30 (IMULT) / (2 (REFDIV) * 3 (ODIV) * 1(SYSDIV))
//
#define DEVICE_SETCLOCK_CFG          (SYSCTL_OSCSRC_XTAL | SYSCTL_IMULT(30) | \
                                      SYSCTL_REFDIV(2) | SYSCTL_ODIV(3) | \
                                      SYSCTL_SYSDIV(1) | SYSCTL_PLL_ENABLE | \
                                      SYSCTL_DCC_BASE_0)

//
// 100MHz SYSCLK frequency based on the above DEVICE_SETCLOCK_CFG. Update the
// code below if a different clock configuration is used!
//
#define DEVICE_SYSCLK_FREQ          ((DEVICE_OSCSRC_FREQ * 30) / (2 * 3 * 1))
#elif defined(CPU_FRQ_120MHZ)

//
// Define to pass to SysCtl_setClock(). Will configure the clock as follows:
// PLLSYSCLK = 20MHz (XTAL_OSC) * 48 (IMULT) / (2 (REFDIV) * 4 (ODIV) * 1(SYSDIV))
//
#define DEVICE_SETCLOCK_CFG          (SYSCTL_OSCSRC_XTAL | SYSCTL_IMULT(48) | \
                                      SYSCTL_REFDIV(2) | SYSCTL_ODIV(4) | \
                                      SYSCTL_SYSDIV(1) | SYSCTL_PLL_ENABLE | \
                                      SYSCTL_DCC_BASE_0)

//
// 120MHz SYSCLK frequency based on the above DEVICE_SETCLOCK_CFG. Update the
// code below if a different clock configuration is used!
//
#define DEVICE_SYSCLK_FREQ          ((DEVICE_OSCSRC_FREQ * 48) / (2 * 4 * 1))
#endif
#elif defined(USE_PLL_SRC_INTOSC)
//
// 10MHz INTOSC on the device is used as the PLL source.
// For use with SysCtl_getClock().
//
#define DEVICE_OSCSRC_FREQ          10000000U
#if defined(CPU_FRQ_100MHZ)
//
// Define to pass to SysCtl_setClock(). Will configure the clock as follows:
// PLLSYSCLK = 10MHz (INT_OSC2) * 30 (IMULT) / (1 (REFDIV) * 3 (ODIV) * 1(SYSDIV))
//
#define DEVICE_SETCLOCK_CFG          (SYSCTL_OSCSRC_OSC2 | SYSCTL_IMULT(30) | \
                                      SYSCTL_REFDIV(1) | SYSCTL_ODIV(3) | \
                                      SYSCTL_SYSDIV(1) | SYSCTL_PLL_ENABLE | \
                                      SYSCTL_DCC_BASE_0)

//
// 100MHz SYSCLK frequency based on the above DEVICE_SETCLOCK_CFG. Update the
// code below if a different clock configuration is used!
//
#define DEVICE_SYSCLK_FREQ          ((DEVICE_OSCSRC_FREQ * 30) / (1 * 3 * 1))
#elif defined(CPU_FRQ_120MHZ)
//
// Define to pass to SysCtl_setClock(). Will configure the clock as follows:
// PLLSYSCLK = 10MHz (INT_OSC2) * 48 (IMULT) / (1 (REFDIV) * 4 (ODIV) * 1(SYSDIV))
//
#define DEVICE_SETCLOCK_CFG          (SYSCTL_OSCSRC_OSC2 | SYSCTL_IMULT(48) | \
                                      SYSCTL_REFDIV(1) | SYSCTL_ODIV(4) | \
                                      SYSCTL_SYSDIV(1) | SYSCTL_PLL_ENABLE | \
                                      SYSCTL_DCC_BASE_0)

//
// 120MHz SYSCLK frequency based on the above DEVICE_SETCLOCK_CFG. Update the
// code below if a different clock configuration is used!
//
#define DEVICE_SYSCLK_FREQ          ((DEVICE_OSCSRC_FREQ * 48) / (1 * 4 * 1))
#endif
#endif

//
// 30MHz (25MHz in case if DEVICE_SYSCLK_FREQ = 100MHz) LSPCLK frequency based on the above DEVICE_SYSCLK_FREQ and a default
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

#define DEVICE_FLASH_WAITSTATES 2

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
