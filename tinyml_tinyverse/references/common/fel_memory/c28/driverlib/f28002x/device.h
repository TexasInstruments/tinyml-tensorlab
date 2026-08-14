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
// Defines for pin numbers
//
//*****************************************************************************
#ifdef _LAUNCHXL_F280025C
//
// LaunchPad
//
//
// LEDs
//
#define DEVICE_GPIO_PIN_LED1        31U             // GPIO number for LED4
#define DEVICE_GPIO_PIN_LED2        34U             // GPIO number for LED5
#define DEVICE_GPIO_CFG_LED1        GPIO_31_GPIO31  // "pinConfig" for LED4
#define DEVICE_GPIO_CFG_LED2        GPIO_34_GPIO34  // "pinConfig" for LED5

//
// CANA
//
#define DEVICE_GPIO_PIN_CANTXA      32U             // GPIO number for CANTXA
#define DEVICE_GPIO_PIN_CANRXA      33U             // GPIO number for CANRXA
#define DEVICE_GPIO_CFG_CANTXA      GPIO_32_CANA_TX  // "pinConfig" for CANA TX
#define DEVICE_GPIO_CFG_CANRXA      GPIO_33_CANA_RX  // "pinConfig" for CANA RX

//
// SCI for USB-to-UART adapter on XDS110 chip
//
#define DEVICE_GPIO_PIN_SCIRXDA     28U             // GPIO number for SCI RX
#define DEVICE_GPIO_PIN_SCITXDA     29U             // GPIO number for SCI TX
#define DEVICE_GPIO_CFG_SCIRXDA     GPIO_28_SCIA_RX // "pinConfig" for SCI RX
#define DEVICE_GPIO_CFG_SCITXDA     GPIO_29_SCIA_TX // "pinConfig" for SCI 

#define DEVICE_GPIO_PIN_SCIRXDA_2   17U             // GPIO number for SCI RX option 2
#define DEVICE_GPIO_PIN_SCITXDA_2   16U             // GPIO number for SCI TX option 2
#define DEVICE_GPIO_CFG_SCIRXDA_2   GPIO_17_SCIA_RX // "pinConfig" for SCI RX option 2
#define DEVICE_GPIO_CFG_SCITXDA_2   GPIO_16_SCIA_TX // "pinConfig" for SCI TX option 2

//
// LINA
//
#define DEVICE_GPIO_PIN_LINTXA      28U               // GPIO number for LINA TX
#define DEVICE_GPIO_PIN_LINRXA      29U               // GPIO number for LINA RX
#define DEVICE_GPIO_CFG_LINTXA      GPIO_28_LINA_TX  // "pinConfig" for LINA TX
#define DEVICE_GPIO_CFG_LINRXA      GPIO_29_LINA_RX  // "pinConfig" for LINA RX

//
// LINB
//
#define DEVICE_GPIO_PIN_LINTXB      12U              // GPIO number for LINB TX
#define DEVICE_GPIO_PIN_LINRXB      13U              // GPIO number for LINB RX
#define DEVICE_GPIO_CFG_LINTXB      GPIO_12_LINB_TX  // "pinConfig" for LINB TX
#define DEVICE_GPIO_CFG_LINRXB      GPIO_13_LINB_RX  // "pinConfig" for LINB RX

//
// I2C
//
#define DEVICE_GPIO_PIN_SDAA        26U  // GPIO number for I2C SDAA
#define DEVICE_GPIO_PIN_SCLA        43U  // GPIO number for I2C SCLA
#define DEVICE_GPIO_CFG_SDAA        GPIO_26_I2CA_SDA  // "pinConfig" for I2C SDAA
#define DEVICE_GPIO_CFG_SCLA        GPIO_43_I2CA_SCL  // "pinConfig" for I2C SCLA

//I2CB GPIO pins
#define DEVICE_GPIO_PIN_SDAB    2
#define DEVICE_GPIO_PIN_SCLB    3

#define DEVICE_GPIO_CFG_SDAB GPIO_2_I2CB_SDA
#define DEVICE_GPIO_CFG_SCLB GPIO_3_I2CB_SCL

//
// FSI
//
#define DEVICE_GPIO_PIN_FSI_RXCLK   13U  // GPIO number for FSI RXCLK
#define DEVICE_GPIO_PIN_FSI_RX0     12U  // GPIO number for FSI RX0
#define DEVICE_GPIO_PIN_FSI_RX1     41U  // GPIO number for FSI RX1
#define DEVICE_GPIO_CFG_FSI_TXCLK   GPIO_7_FSITXA_CLK  // "pinConfig" for FSI TXCLK
#define DEVICE_GPIO_CFG_FSI_TX0     GPIO_6_FSITXA_D0   // "pinConfig" for FSI TX0
#define DEVICE_GPIO_CFG_FSI_TX1     GPIO_46_FSITXA_D1  // "pinConfig" for FSI TX1
#define DEVICE_GPIO_CFG_FSI_RXCLK   GPIO_13_FSIRXA_CLK // "pinConfig" for FSI RXCLK
#define DEVICE_GPIO_CFG_FSI_RX0     GPIO_12_FSIRXA_D0  // "pinConfig" for FSI RX0
#define DEVICE_GPIO_CFG_FSI_RX1     GPIO_41_FSIRXA_D1  // "pinConfig" for FSI RX1

//
// SPIA
//
#define DEVICE_GPIO_PIN_SPICLKA     9U   // GPIO number for SPI CLKA
#define DEVICE_GPIO_PIN_SPISIMOA    11U  // GPIO number for SPI SIMOA
#define DEVICE_GPIO_PIN_SPISOMIA    10U  // GPIO number for SPI SOMIA
#define DEVICE_GPIO_PIN_SPISTEA     5U   // GPIO number for SPI STEA
#define DEVICE_GPIO_CFG_SPICLKA     GPIO_9_SPIA_CLK    // "pinConfig" for SPI CLKA
#define DEVICE_GPIO_CFG_SPISIMOA    GPIO_11_SPIA_SIMO  // "pinConfig" for SPI SIMOA
#define DEVICE_GPIO_CFG_SPISOMIA    GPIO_10_SPIA_SOMI  // "pinConfig" for SPI SOMIA
#define DEVICE_GPIO_CFG_SPISTEA     GPIO_5_SPIA_STE    // "pinConfig" for SPI STEA

//
// SPIB
//
#define DEVICE_GPIO_PIN_SPICLKB     22U  // GPIO number for SPI CLKB
#define DEVICE_GPIO_PIN_SPISIMOB    30U  // GPIO number for SPI SIMOB
#define DEVICE_GPIO_PIN_SPISOMIB    31U  // GPIO number for SPI SOMIB
#define DEVICE_GPIO_PIN_SPISTEB     33U  // GPIO number for SPI STEB
#define DEVICE_GPIO_CFG_SPICLKB     GPIO_22_SPIB_CLK   // "pinConfig" for SPI CLKB
#define DEVICE_GPIO_CFG_SPISIMOB    GPIO_30_SPIB_SIMO  // "pinConfig" for SPI SIMOB
#define DEVICE_GPIO_CFG_SPISOMIB    GPIO_31_SPIB_SOMI  // "pinConfig" for SPI SOMIB
#define DEVICE_GPIO_CFG_SPISTEB     GPIO_33_SPIB_STE   // "pinConfig" for SPI STEB

//
// eQEP1
//
#define DEVICE_GPIO_PIN_EQEP1A      44U  // GPIO number for EQEP 1A
#define DEVICE_GPIO_PIN_EQEP1B      37U  // GPIO number for EQEP 1B
#define DEVICE_GPIO_PIN_EQEP1I      43U  // GPIO number for EQEP 1I
#define DEVICE_GPIO_CFG_EQEP1A      GPIO_44_EQEP1_A  // "pinConfig" for EQEP 1A
#define DEVICE_GPIO_CFG_EQEP1B      GPIO_37_EQEP1_B  // "pinConfig" for EQEP 1B
#define DEVICE_GPIO_CFG_EQEP1I      GPIO_43_EQEP1_INDEX  // "pinConfig" for EQEP 1I

//
// eQEP2
//
#define DEVICE_GPIO_PIN_EQEP2A      14U  // GPIO number for EQEP 2A
#define DEVICE_GPIO_PIN_EQEP2B      25U  // GPIO number for EQEP 2B
#define DEVICE_GPIO_PIN_EQEP2I      26U  // GPIO number for EQEP 2I
#define DEVICE_GPIO_CFG_EQEP2A      GPIO_14_EQEP2_A  // "pinConfig" for EQEP 2A
#define DEVICE_GPIO_CFG_EQEP2B      GPIO_25_EQEP2_B  // "pinConfig" for EQEP 2B
#define DEVICE_GPIO_CFG_EQEP2I      GPIO_26_EQEP2_INDEX  // "pinConfig" for EQEP 2I

#else
//
// ControlCARD
//

//
// LEDs
//
#define DEVICE_GPIO_PIN_LED1        31U             // GPIO number for LED1
#define DEVICE_GPIO_PIN_LED2        34U             // GPIO number for LED2
#define DEVICE_GPIO_CFG_LED1        GPIO_31_GPIO31  // "pinConfig" for LED1
#define DEVICE_GPIO_CFG_LED2        GPIO_34_GPIO34  // "pinConfig" for LED2

//
// CANA
//
#define DEVICE_GPIO_PIN_CANTXA      31U             // GPIO number for CANTXA
#define DEVICE_GPIO_PIN_CANRXA      30U             // GPIO number for CANRXA
#define DEVICE_GPIO_CFG_CANRXA      GPIO_30_CANA_RX  // "pinConfig" for CANA RX
#define DEVICE_GPIO_CFG_CANTXA      GPIO_31_CANA_TX  // "pinConfig" for CANA TX

//
// SCI for USB-to-UART adapter on FTDI chip
//
#define DEVICE_GPIO_PIN_SCIRXDA     28U             // GPIO number for SCI RX
#define DEVICE_GPIO_PIN_SCITXDA     29U             // GPIO number for SCI TX
#define DEVICE_GPIO_CFG_SCIRXDA     GPIO_28_SCIA_RX // "pinConfig" for SCI RX
#define DEVICE_GPIO_CFG_SCITXDA     GPIO_29_SCIA_TX // "pinConfig" for SCI TX

//
// LINA
//
#define DEVICE_GPIO_PIN_LINTXA      22U               // GPIO number for LINA TX
#define DEVICE_GPIO_PIN_LINRXA      23U               // GPIO number for LINA RX
#define DEVICE_GPIO_CFG_LINTXA      GPIO_22_LINA_TX  // "pinConfig" for LINA TX
#define DEVICE_GPIO_CFG_LINRXA      GPIO_23_LINA_RX  // "pinConfig" for LINA RX

//
// LINB
//
#define DEVICE_GPIO_PIN_LINTXB      40U              // GPIO number for LINB TX
#define DEVICE_GPIO_PIN_LINRXB      41U              // GPIO number for LINB RX
#define DEVICE_GPIO_CFG_LINTXB      GPIO_40_LINB_TX  // "pinConfig" for LINB TX
#define DEVICE_GPIO_CFG_LINRXB      GPIO_41_LINB_RX  // "pinConfig" for LINB RX

//
// I2C
//
#define DEVICE_GPIO_PIN_SDAA        26U  // GPIO number for I2C SDAA
#define DEVICE_GPIO_PIN_SCLA        27U  // GPIO number for I2C SCLA
#define DEVICE_GPIO_CFG_SDAA        GPIO_26_I2CA_SDA  // "pinConfig" for I2C SDAA
#define DEVICE_GPIO_CFG_SCLA        GPIO_27_I2CA_SCL  // "pinConfig" for I2C SCLA

//I2CB GPIO pins
#define DEVICE_GPIO_PIN_SDAB    2
#define DEVICE_GPIO_PIN_SCLB    3

#define DEVICE_GPIO_CFG_SDAB GPIO_2_I2CB_SDA
#define DEVICE_GPIO_CFG_SCLB GPIO_3_I2CB_SCL

//
// FSI
//
#define DEVICE_GPIO_PIN_FSI_RXCLK   13U  // GPIO number for FSI RXCLK
#define DEVICE_GPIO_PIN_FSI_RX0     12U  // GPIO number for FSI RX0
#define DEVICE_GPIO_PIN_FSI_RX1     11U  // GPIO number for FSI RX1
#define DEVICE_GPIO_CFG_FSI_TXCLK   GPIO_27_FSITXA_CLK  // "pinConfig" for FSI TXCLK
#define DEVICE_GPIO_CFG_FSI_TX0     GPIO_26_FSITXA_D0   // "pinConfig" for FSI TX0
#define DEVICE_GPIO_CFG_FSI_TX1     GPIO_25_FSITXA_D1   // "pinConfig" for FSI TX1
#define DEVICE_GPIO_CFG_FSI_RXCLK   GPIO_13_FSIRXA_CLK  // "pinConfig" for FSI RXCLK
#define DEVICE_GPIO_CFG_FSI_RX0     GPIO_12_FSIRXA_D0   // "pinConfig" for FSI RX0
#define DEVICE_GPIO_CFG_FSI_RX1     GPIO_11_FSIRXA_D1   // "pinConfig" for FSI RX1

#endif

//*****************************************************************************
//
// Defines related to clock configuration
//
//*****************************************************************************

//
// To use INTOSC as the clock source, comment the #define USE_PLL_SRC_XTAL,
// and uncomment the #define USE_PLL_SRC_INTOSC
//
#define USE_PLL_SRC_XTAL
//#define USE_PLL_SRC_INTOSC

#if defined(USE_PLL_SRC_XTAL)
//
// 20MHz XTAL on controlCARD is used as the PLL source.
// For use with SysCtl_getClock().
//
#define DEVICE_OSCSRC_FREQ          20000000U

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

#elif defined(USE_PLL_SRC_INTOSC)
//
// 10MHz INTOSC on the device is used as the PLL source.
// For use with SysCtl_getClock().
//
#define DEVICE_OSCSRC_FREQ          10000000U

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
#endif

//
// 25MHz LSPCLK frequency based on the above DEVICE_SYSCLK_FREQ and a default
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

//
//  Defines for setting FSI clock speeds
//
#define FSI_PRESCALE_50MHZ          1U
#define FSI_PRESCALE_25MHZ          2U
#define FSI_PRESCALE_10MHZ          5U
#define FSI_PRESCALE_5MHZ           10U

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

#define DEVICE_FLASH_WAITSTATES 4

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
