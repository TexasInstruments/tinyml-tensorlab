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
// Defines for pin numbers and other GPIO configuration
//
//*****************************************************************************
#ifdef _LAUNCHXL_F280049C
//
// LaunchPad
//

//
// LEDs
//
#define DEVICE_GPIO_PIN_LED1        23U  // GPIO number for LD4
#define DEVICE_GPIO_PIN_LED2        34U  // GPIO number for LD5
#define DEVICE_GPIO_CFG_LED1        GPIO_23_GPIO23  // "pinConfig" for LD4
#define DEVICE_GPIO_CFG_LED2        GPIO_34_GPIO34  // "pinConfig" for LD5

//
// CANA
//
#define DEVICE_GPIO_PIN_CANTXA      32U  // GPIO number for CANTXA
#define DEVICE_GPIO_PIN_CANRXA      33U  // GPIO number for CANRXA
#define DEVICE_GPIO_CFG_CANRXA      GPIO_33_CANA_RX  // "pinConfig" for CANA RX
#define DEVICE_GPIO_CFG_CANTXA      GPIO_32_CANA_TX  // "pinConfig" for CANA TX

//
// CAN External Loopback
//
#define DEVICE_GPIO_CFG_CANRXB      GPIO_10_CANB_RX  // "pinConfig" for CANB RX
#define DEVICE_GPIO_CFG_CANTXB      GPIO_8_CANB_TX   // "pinConfig" for CANB TX


//
// LINA
//
#define DEVICE_GPIO_CFG_LINTXA      GPIO_37_LINA_TX  // "pinConfig" for LINA TX
#define DEVICE_GPIO_CFG_LINRXA      GPIO_35_LINA_RX  // "pinConfig" for LINA RX

//
// FSI
//
#define DEVICE_GPIO_PIN_FSI_RXCLK   33U  // GPIO number for FSI RXCLK
#define DEVICE_GPIO_PIN_FSI_RX0     12U  // GPIO number for FSI RX0
#define DEVICE_GPIO_PIN_FSI_RX1      2U  // GPIO number for FSI RX1
#define DEVICE_GPIO_CFG_FSI_TXCLK   GPIO_7_FSITXA_CLK  // "pinConfig" for FSI TXCLK
#define DEVICE_GPIO_CFG_FSI_TX0     GPIO_6_FSITXA_D0  // "pinConfig" for FSI TX0
#define DEVICE_GPIO_CFG_FSI_TX1     GPIO_25_FSITXA_D1  // "pinConfig" for FSI TX1
#define DEVICE_GPIO_CFG_FSI_RXCLK   GPIO_33_FSIRXA_CLK  // "pinConfig" for FSI RXCLK
#define DEVICE_GPIO_CFG_FSI_RX0     GPIO_12_FSIRXA_D0  // "pinConfig" for FSI RX0
#define DEVICE_GPIO_CFG_FSI_RX1     GPIO_2_FSIRXA_D1  // "pinConfig" for FSI RX1

//
// SPI
//
#define DEVICE_GPIO_PIN_SPICLKA     56U  // GPIO number for SPI CLKA
#define DEVICE_GPIO_PIN_SPISIMOA    16U  // GPIO number for SPI SIMOA
#define DEVICE_GPIO_PIN_SPISOMIA    17U  // GPIO number for SPI SOMIA
#define DEVICE_GPIO_PIN_SPISTEA     57U  // GPIO number for SPI STEA
#define DEVICE_GPIO_CFG_SPICLKA     GPIO_56_SPIA_CLK   // "pinConfig" for SPI CLKA
#define DEVICE_GPIO_CFG_SPISIMOA    GPIO_16_SPIA_SIMO  // "pinConfig" for SPI SIMOA
#define DEVICE_GPIO_CFG_SPISOMIA    GPIO_17_SPIA_SOMI  // "pinConfig" for SPI SOMIA
#define DEVICE_GPIO_CFG_SPISTEA     GPIO_57_SPIA_STE   // "pinConfig" for SPI STEA
#define DEVICE_GPIO_PIN_SPICLKB     22U  // GPIO number for SPI CLKB
#define DEVICE_GPIO_PIN_SPISIMOB    24U  // GPIO number for SPI SIMOB
#define DEVICE_GPIO_PIN_SPISOMIB    31U  // GPIO number for SPI SOMIB
#define DEVICE_GPIO_PIN_SPISTEB     27U  // GPIO number for SPI STEB
#define DEVICE_GPIO_CFG_SPICLKB     GPIO_22_SPIB_CLK   // "pinConfig" for SPI CLKB
#define DEVICE_GPIO_CFG_SPISIMOB    GPIO_24_SPIB_SIMO  // "pinConfig" for SPI SIMOB
#define DEVICE_GPIO_CFG_SPISOMIB    GPIO_31_SPIB_SOMI  // "pinConfig" for SPI SOMIB
#define DEVICE_GPIO_CFG_SPISTEB     GPIO_27_SPIB_STE   // "pinConfig" for SPI STEB

//
// I2C
//
#define DEVICE_GPIO_PIN_SDAA        35U  // GPIO number for I2C SDAA
#define DEVICE_GPIO_PIN_SCLA        37U  // GPIO number for I2C SCLA
#define DEVICE_GPIO_CFG_SDAA        GPIO_35_I2CA_SDA  // "pinConfig" for I2C SDAA
#define DEVICE_GPIO_CFG_SCLA        GPIO_37_I2CA_SCL  // "pinConfig" for I2C SCLA

//
// eQEP
//
#define DEVICE_GPIO_PIN_EQEP1A      35U  // GPIO number for EQEP 1A
#define DEVICE_GPIO_PIN_EQEP1B      37U  // GPIO number for EQEP 1B
#define DEVICE_GPIO_PIN_EQEP1I      59U  // GPIO number for EQEP 1I
#define DEVICE_GPIO_CFG_EQEP1A      GPIO_35_EQEP1_A  // "pinConfig" for EQEP 1A
#define DEVICE_GPIO_CFG_EQEP1B      GPIO_37_EQEP1_B  // "pinConfig" for EQEP 1B
#define DEVICE_GPIO_CFG_EQEP1I      GPIO_59_EQEP1_INDEX  // "pinConfig" for EQEP 1I

//
// EPWM
//
#define DEVICE_PERIPHERAL_BASE_EPWM EPWM6_BASE  // Base peripheral EPWM 6
#define DEVICE_PERIPHERAL_INT_EPWM  INT_EPWM6   // Base peripheral EPWM 6 interrupt
#define DEVICE_GPIO_PIN_EPWMxA      10U  // GPIO number for EPWM6A
#define DEVICE_GPIO_PIN_EPWMxB      11U  // GPIO number for EPWM6B
#define DEVICE_GPIO_PIN_IOINDEX     8U   // GPIO number for IO Index
#define DEVICE_GPIO_CFG_EPWMxA      GPIO_10_EPWM6_A  // "pinConfig" for EPWM6A
#define DEVICE_GPIO_CFG_EPWMxB      GPIO_11_EPWM6_B  // "pinConfig" for EPWM6B
#define DEVICE_GPIO_CFG_IOINDEX     GPIO_8_GPIO8    // "pinConfig" for IO Index

#else
//
// ControlCARD
//    

//
// LEDs
//
#define DEVICE_GPIO_PIN_LED1        31U  // GPIO number for LD2
#define DEVICE_GPIO_PIN_LED2        34U  // GPIO number for LD3
#define DEVICE_GPIO_CFG_LED1        GPIO_31_GPIO31  // "pinConfig" for LD2
#define DEVICE_GPIO_CFG_LED2        GPIO_34_GPIO34  // "pinConfig" for LD3

//
// CANA
//
#define DEVICE_GPIO_PIN_CANTXA      31U  // GPIO number for CANTXA
#define DEVICE_GPIO_PIN_CANRXA      30U  // GPIO number for CANRXA

//
// CAN External Loopback
//
#define DEVICE_GPIO_CFG_CANRXA      GPIO_30_CANA_RX  // "pinConfig" for CANA RX
#define DEVICE_GPIO_CFG_CANTXA      GPIO_31_CANA_TX  // "pinConfig" for CANA TX
#define DEVICE_GPIO_CFG_CANRXB      GPIO_10_CANB_RX  // "pinConfig" for CANB RX
#define DEVICE_GPIO_CFG_CANTXB      GPIO_8_CANB_TX   // "pinConfig" for CANB TX

//
// LINA
//
#define DEVICE_GPIO_CFG_LINTXA      GPIO_58_LINA_TX  // "pinConfig" for LINA TX
#define DEVICE_GPIO_CFG_LINRXA      GPIO_59_LINA_RX  // "pinConfig" for LINA RX

//
// FSI
//
#define DEVICE_GPIO_PIN_FSI_RXCLK   13U  // GPIO number for FSI RXCLK
#define DEVICE_GPIO_PIN_FSI_RX0     12U  // GPIO number for FSI RX0
#define DEVICE_GPIO_PIN_FSI_RX1     11U  // GPIO number for FSI RX1
#define DEVICE_GPIO_CFG_FSI_TXCLK   GPIO_27_FSITXA_CLK  // "pinConfig" for FSI TXCLK
#define DEVICE_GPIO_CFG_FSI_TX0     GPIO_26_FSITXA_D0  // "pinConfig" for FSI TX0
#define DEVICE_GPIO_CFG_FSI_TX1     GPIO_25_FSITXA_D1  // "pinConfig" for FSI TX1
#define DEVICE_GPIO_CFG_FSI_RXCLK   GPIO_13_FSIRXA_CLK  // "pinConfig" for FSI RXCLK
#define DEVICE_GPIO_CFG_FSI_RX0     GPIO_12_FSIRXA_D0  // "pinConfig" for FSI RX0
#define DEVICE_GPIO_CFG_FSI_RX1     GPIO_11_FSIRXA_D1  // "pinConfig" for FSI RX1

//
// SPI
//
#define DEVICE_GPIO_PIN_SPICLKA     9U  // GPIO number for SPI CLKA
#define DEVICE_GPIO_PIN_SPISIMOA    8U  // GPIO number for SPI SIMOA
#define DEVICE_GPIO_PIN_SPISOMIA    10U  // GPIO number for SPI SOMIA
#define DEVICE_GPIO_PIN_SPISTEA     11U  // GPIO number for SPI STEA
#define DEVICE_GPIO_CFG_SPICLKA     GPIO_9_SPIA_CLK   // "pinConfig" for SPI CLKA
#define DEVICE_GPIO_CFG_SPISIMOA    GPIO_8_SPIA_SIMO  // "pinConfig" for SPI SIMOA
#define DEVICE_GPIO_CFG_SPISOMIA    GPIO_10_SPIA_SOMI // "pinConfig" for SPI SOMIA
#define DEVICE_GPIO_CFG_SPISTEA     GPIO_11_SPIA_STE   // "pinConfig" for SPI STEA
#define DEVICE_GPIO_PIN_SPICLKB     26U  // GPIO number for SPI CLKB
#define DEVICE_GPIO_PIN_SPISIMOB    24U  // GPIO number for SPI SIMOB
#define DEVICE_GPIO_PIN_SPISOMIB    25U  // GPIO number for SPI SOMIB
#define DEVICE_GPIO_PIN_SPISTEB     27U  // GPIO number for SPI STEB
#define DEVICE_GPIO_CFG_SPICLKB     GPIO_26_SPIB_CLK   // "pinConfig" for SPI CLKB
#define DEVICE_GPIO_CFG_SPISIMOB    GPIO_24_SPIB_SIMO  // "pinConfig" for SPI SIMOB
#define DEVICE_GPIO_CFG_SPISOMIB    GPIO_25_SPIB_SOMI  // "pinConfig" for SPI SOMIB
#define DEVICE_GPIO_CFG_SPISTEB     GPIO_27_SPIB_STE   // "pinConfig" for SPI STEB

//
// I2C
//
#define DEVICE_GPIO_PIN_SDAA        32U  // GPIO number for I2C SDAA
#define DEVICE_GPIO_PIN_SCLA        33U  // GPIO number for I2C SCLA
#define DEVICE_GPIO_CFG_SDAA        GPIO_32_I2CA_SDA  // "pinConfig" for I2C SDAA
#define DEVICE_GPIO_CFG_SCLA        GPIO_33_I2CA_SCL  // "pinConfig" for I2C SCLA

//
// eQEP
//
#define DEVICE_GPIO_PIN_EQEP1A      6U  // GPIO number for EQEP 1A
#define DEVICE_GPIO_PIN_EQEP1B      7U  // GPIO number for EQEP 1B
#define DEVICE_GPIO_PIN_EQEP1I      9U  // GPIO number for EQEP 1I
#define DEVICE_GPIO_CFG_EQEP1A      GPIO_6_EQEP1_A  // "pinConfig" for EQEP 1A
#define DEVICE_GPIO_CFG_EQEP1B      GPIO_7_EQEP1_B  // "pinConfig" for EQEP 1B
#define DEVICE_GPIO_CFG_EQEP1I      GPIO_9_EQEP1_INDEX  // "pinConfig" for EQEP 1I

//
// EPWM
//
#define DEVICE_PERIPHERAL_BASE_EPWM EPWM1_BASE  // Base peripheral EPWM 1
#define DEVICE_PERIPHERAL_INT_EPWM  INT_EPWM1   // Base peripheral EPWM 1 interrupt
#define DEVICE_GPIO_PIN_EPWMxA      0U  // GPIO number for EPWM1A
#define DEVICE_GPIO_PIN_EPWMxB      1U  // GPIO number for EPWM1B
#define DEVICE_GPIO_PIN_IOINDEX     2U  // GPIO number for IO Index
#define DEVICE_GPIO_CFG_EPWMxA      GPIO_0_EPWM1_A  // "pinConfig" for EPWM1A
#define DEVICE_GPIO_CFG_EPWMxB      GPIO_1_EPWM1_B  // "pinConfig" for EPWM1B
#define DEVICE_GPIO_CFG_IOINDEX     GPIO_2_GPIO2   // "pinConfig" for IO Index
#endif

#define DEVICE_GPIO_CFG_EPWM1A      GPIO_0_EPWM1_A  // "pinConfig" for EPWM1A
#define DEVICE_GPIO_PIN_EPWM1A      0U              // GPIO number for EPWM1A
//
// SCI for USB-to-UART adapter on FTDI chip
//
#define DEVICE_GPIO_PIN_SCIRXDA     28U             // GPIO number for SCI RX
#define DEVICE_GPIO_PIN_SCITXDA     29U             // GPIO number for SCI TX
#define DEVICE_GPIO_CFG_SCIRXDA     GPIO_28_SCIA_RX // "pinConfig" for SCI RX
#define DEVICE_GPIO_CFG_SCITXDA     GPIO_29_SCIA_TX // "pinConfig" for SCI TX

//*****************************************************************************
//
// Defines related to clock configuration
//
//*****************************************************************************
//
// 20MHz XTAL on controlCARD and Launchpad. For use with SysCtl_getClock().
//
#define DEVICE_OSCSRC_FREQ          20000000U

//
// Define to pass to SysCtl_setClock(). Will configure the clock as follows:
// PLLSYSCLK = 20MHz (XTAL_OSC) * 10 (IMULT) * 1 (FMULT) / 2 (PLLCLK_BY_2)
//
#define DEVICE_SETCLOCK_CFG         (SYSCTL_OSCSRC_XTAL | SYSCTL_IMULT(10) |  \
                                     SYSCTL_FMULT_NONE | SYSCTL_SYSDIV(2) |   \
                                     SYSCTL_PLL_ENABLE)

//
// 100MHz SYSCLK frequency based on the above DEVICE_SETCLOCK_CFG. Update the
// code below if a different clock configuration is used!
//
#define DEVICE_SYSCLK_FREQ          ((DEVICE_OSCSRC_FREQ * 10 * 1) / 2)

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

//*****************************************************************************
//  Defines for setting FSI clock speeds in normal FSI mode (SPI_MODE == 0)
//*****************************************************************************
//
//  TXCLKIN = INPUT_CLOCK / PRESCALE_VAL
//  TXCLK (in FSI mode) = TXCLKIN / 2
//  TXCLK (in SPI mode) = TXCLKIN
//
//  Below assumes INPUT_CLOCK == 200 MHz (PLLRAWCLK, SEL_PLLCLK == 1)
//
#define FSI_PRESCALE_50MHZ          2U
#define FSI_PRESCALE_25MHZ          4U
#define FSI_PRESCALE_10MHZ          10U
#define FSI_PRESCALE_5MHZ           20U

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


extern uint32_t Example_PassCount;
extern uint32_t Example_Fail;

//*****************************************************************************
//
// Function Prototypes
//
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//
//! \addtogroup device_api
//! @{
//
//***
//*****************************************************************************
//
//! @brief Function to initialize the device. Primarily initializes system control to a
//! known state by disabling the watchdog, setting up the SYSCLKOUT frequency,
//! and enabling the clocks to the peripherals.
//! The function also configures the GPIO pins 22 and 23 in digital mode.
//! To configure these pins as analog pins, use the function GPIO_setAnalogMode
//!
//! \return None
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
//! \return None
//
//*****************************************************************************
extern void Device_enableAllPeripherals(void);

//*****************************************************************************
//!
//!
//! @brief Function to disable pin locks and enable pullups on GPIOs.
//!
//! \return None
//
//*****************************************************************************
extern void Device_initGPIO(void);

//*****************************************************************************
//!
//! @brief Error handling function to be called when an ASSERT is violated
//!
//! \param *filename File name in which the error has occurred
//! \param line Line number within the file
//! \return None
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
