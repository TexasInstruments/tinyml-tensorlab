//#############################################################################
//
// FILE:   device.h
//
// TITLE:  Device setup for examples.
//
//#############################################################################
//
// $Release Date: $
// $Copyright:
// Copyright (C) 2013-2026 Texas Instruments Incorporated - http://www.ti.com/
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

//
// Included Files
//
#include "driverlib.h"

#if (!defined(CPU1) && !defined(CPU2))
#error "You must define CPU1 or CPU2 in your project properties.  Otherwise, \
the offsets in your header files will be inaccurate."
#endif

#if (defined(CPU1) && defined(CPU2))
#error "You have defined both CPU1 and CPU2 in your project properties.  Only \
a single CPU should be defined."
#endif

//*****************************************************************************
//
// Defines for pin numbers and other GPIO configuration
//
//*****************************************************************************
//
// LEDs
//
#ifdef _LAUNCHXL_F28379D
#define DEVICE_GPIO_PIN_LED1        31U  // GPIO number for LD10
#define DEVICE_GPIO_PIN_LED2        34U  // GPIO number for LD9
#define DEVICE_GPIO_CFG_LED1        GPIO_31_GPIO31  // "pinConfig" for LD10
#define DEVICE_GPIO_CFG_LED2        GPIO_34_GPIO34  // "pinConfig" for LD9
#else
#define DEVICE_GPIO_PIN_LED1        31U  // GPIO number for LD2
#define DEVICE_GPIO_PIN_LED2        34U  // GPIO number for LD3
#define DEVICE_GPIO_CFG_LED1        GPIO_31_GPIO31  // "pinConfig" for LD2
#define DEVICE_GPIO_CFG_LED2        GPIO_34_GPIO34  // "pinConfig" for LD3
#endif


//
// SCI for USB-to-UART adapter on FTDI chip
//
#ifdef _LAUNCHXL_F28379D
#define DEVICE_GPIO_PIN_SCIRXDA     43U             // GPIO number for SCI RX
#define DEVICE_GPIO_PIN_SCITXDA     42U             // GPIO number for SCI TX
#define DEVICE_GPIO_CFG_SCIRXDA     GPIO_43_SCIRXDA // "pinConfig" for SCI RX
#define DEVICE_GPIO_CFG_SCITXDA     GPIO_42_SCITXDA // "pinConfig" for SCI TX
#else
#define DEVICE_GPIO_PIN_SCIRXDA     28U             // GPIO number for SCI RX
#define DEVICE_GPIO_PIN_SCITXDA     29U             // GPIO number for SCI TX
#define DEVICE_GPIO_CFG_SCIRXDA     GPIO_28_SCIRXDA // "pinConfig" for SCI RX
#define DEVICE_GPIO_CFG_SCITXDA     GPIO_29_SCITXDA // "pinConfig" for SCI TX
#endif

//
// GPIO assignment for CAN-A and CAN-B
//
#ifdef _LAUNCHXL_F28379D
#define DEVICE_GPIO_CFG_CANRXA      GPIO_36_CANRXA  // "pinConfig" for CANA RX
#define DEVICE_GPIO_CFG_CANTXA      GPIO_37_CANTXA  // "pinConfig" for CANA TX
#define DEVICE_GPIO_CFG_CANRXB      GPIO_17_CANRXB  // "pinConfig" for CANB RX
#define DEVICE_GPIO_CFG_CANTXB      GPIO_12_CANTXB  // "pinConfig" for CANB TX
#else
#define DEVICE_GPIO_CFG_CANRXA      GPIO_30_CANRXA  // "pinConfig" for CANA RX
#define DEVICE_GPIO_CFG_CANTXA      GPIO_31_CANTXA  // "pinConfig" for CANA TX
#define DEVICE_GPIO_CFG_CANRXB      GPIO_10_CANRXB  // "pinConfig" for CANB RX
#define DEVICE_GPIO_CFG_CANTXB      GPIO_8_CANTXB   // "pinConfig" for CANB TX

//I2CA GPIO pins
#define DEVICE_GPIO_PIN_SDAA    104
#define DEVICE_GPIO_PIN_SCLA    105

#define DEVICE_GPIO_CFG_SDAA GPIO_104_SDAA
#define DEVICE_GPIO_CFG_SCLA GPIO_105_SCLA


//I2CB GPIO pins
#define DEVICE_GPIO_PIN_SDAB    40
#define DEVICE_GPIO_PIN_SCLB    41

#define DEVICE_GPIO_CFG_SDAB GPIO_40_SDAB
#define DEVICE_GPIO_CFG_SCLB GPIO_41_SCLB

#endif

//*****************************************************************************
//
// Defines related to clock configuration
//
//*****************************************************************************
//
// Launchpad Configuration
//
#ifdef _LAUNCHXL_F28379D

//
// 10MHz XTAL on LaunchPad. For use with SysCtl_getClock().
//
#define DEVICE_OSCSRC_FREQ          10000000U

//
// Define to pass to SysCtl_setClock(). Will configure the clock as follows:
// PLLSYSCLK = 10MHz (XTAL_OSC) * 40 (IMULT) * 1 (FMULT) / 2 (PLLCLK_BY_2)
//
#define DEVICE_SETCLOCK_CFG         (SYSCTL_OSCSRC_XTAL | SYSCTL_IMULT(40) |  \
                                     SYSCTL_FMULT_NONE | SYSCTL_SYSDIV(2) |   \
                                     SYSCTL_PLL_ENABLE)

//
// 200MHz SYSCLK frequency based on the above DEVICE_SETCLOCK_CFG. Update the
// code below if a different clock configuration is used!
//
#define DEVICE_SYSCLK_FREQ          ((DEVICE_OSCSRC_FREQ * 40 * 1) / 2)

//
// ControlCARD Configuration
//
#else

//
// 20MHz XTAL on controlCARD. For use with SysCtl_getClock().
//
#define DEVICE_OSCSRC_FREQ          20000000U

//
// Define to pass to SysCtl_setClock(). Will configure the clock as follows:
// PLLSYSCLK = 20MHz (XTAL_OSC) * 20 (IMULT) * 1 (FMULT) / 2 (PLLCLK_BY_2)
//
#define DEVICE_SETCLOCK_CFG         (SYSCTL_OSCSRC_XTAL | SYSCTL_IMULT(20) |  \
                                     SYSCTL_FMULT_NONE | SYSCTL_SYSDIV(2) |   \
                                     SYSCTL_PLL_ENABLE)

//
// 200MHz SYSCLK frequency based on the above DEVICE_SETCLOCK_CFG. Update the
// code below if a different clock configuration is used!
//
#define DEVICE_SYSCLK_FREQ          ((DEVICE_OSCSRC_FREQ * 20 * 1) / 2)

#endif

//
// 50MHz LSPCLK frequency based on the above DEVICE_SYSCLK_FREQ and a default
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
// The macros that can be used as parameter to the function Device_bootCPU2
//
#define C1C2_BROM_BOOTMODE_BOOT_FROM_PARALLEL                        0x00000000U
#define C1C2_BROM_BOOTMODE_BOOT_FROM_SCI                             0x00000001U
#define C1C2_BROM_BOOTMODE_BOOT_FROM_SPI                             0x00000004U
#define C1C2_BROM_BOOTMODE_BOOT_FROM_I2C                             0x00000005U
#define C1C2_BROM_BOOTMODE_BOOT_FROM_CAN                             0x00000007U
#define C1C2_BROM_BOOTMODE_BOOT_FROM_RAM                             0x0000000AU
#define C1C2_BROM_BOOTMODE_BOOT_FROM_FLASH                           0x0000000BU

//
// Other macros that are needed for the Device_bootCPU2 function
//
#define BROM_IPC_EXECUTE_BOOTMODE_CMD                                0x00000013U
#define C1C2_BROM_BOOTMODE_BOOT_COMMAND_MAX_SUPPORT_VALUE            0x0000000CU
#define C2_BOOTROM_BOOTSTS_C2TOC1_IGNORE                             0x00000000U
#define C2_BOOTROM_BOOTSTS_SYSTEM_START_BOOT                         0x00000001U
#define C2_BOOTROM_BOOTSTS_SYSTEM_READY                              0x00000002U
#define C2_BOOTROM_BOOTSTS_C2TOC1_BOOT_CMD_ACK                       0x00000003U
#define C2_BOOTROM_BOOTSTS_C2TOC1_BOOT_CMD_NAK_STATUS_NOT_SUPPORTED  0x00000004U
#define C2_BOOTROM_BOOTSTS_C2TOC1_BOOT_CMD_NAK_STATUS_BUSY_WITH_BOOT 0x00000005U

//
// Macros used as return value by the Device_bootCPU2 function
//
#define STATUS_FAIL                 0x0001
#define STATUS_PASS                 0x0000

//*****************************************************************************
//
// Defines, Globals, and Header Includes related to Flash Support
//
//*****************************************************************************
#ifdef _FLASH
#include <stddef.h>

extern uint16_t RamfuncsLoadStart;
extern uint16_t RamfuncsLoadEnd;
extern uint16_t RamfuncsLoadSize;
extern uint16_t RamfuncsRunStart;
extern uint16_t RamfuncsRunEnd;
extern uint16_t RamfuncsRunSize;

#define DEVICE_FLASH_WAITSTATES 3

#endif

extern uint32_t Example_PassCount;
extern uint32_t Example_Fail;

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
//! @brief Function to initialize the device. Primarily initializes system control to a
//! known state by disabling the watchdog, setting up the SYSCLKOUT frequency,
//! and enabling the clocks to the peripherals.
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
//! @brief Function to enable pullups for the unbonded GPIOs on the 176PTP package:
//! GPIOs     Grp Bits
//! 95-132    C   31
//!           D   31:0
//!           E   4:0
//! 134-168   E   31:6
//!           F   8:0
//!
//! @param None
//! @return None
//
//*****************************************************************************
extern void Device_enableUnbondedGPIOPullupsFor176Pin(void);
//*****************************************************************************
//!
//! @brief Function to enable pullups for the unbonded GPIOs on the 100PZ package:
//! GPIOs     Grp Bits
//! 0-1       A   1:0
//! 5-9       A   9:5
//! 22-40     A   31:22
//!           B   8:0
//! 44-57     B   25:12
//! 67-68     C   4:3
//! 74-77     C   13:10
//! 79-83     C   19:15
//! 93-168    C   31:29
//!           D   31:0
//!           E   31:0
//!           F   8:0
//! @param None
//! @return None
//
//
//*****************************************************************************
extern void Device_enableUnbondedGPIOPullupsFor100Pin(void);
//*****************************************************************************
//!
//! @brief Function to enable pullups for the unbonded GPIOs on the
//! 176PTP package.
//!
//! @param None
//! @return None
//
//*****************************************************************************
extern void Device_enableUnbondedGPIOPullups(void);
#ifdef CPU1
//*****************************************************************************
//!
//! @brief Function to implement Analog trim of TMX devices
//!
//! @param None
//! @return None
//
//*****************************************************************************
extern void Device_configureTMXAnalogTrim(void);
//*****************************************************************************
//! @brief Executes a CPU02 control system bootloader.
//!
//! \param bootMode specifies which CPU02 control system boot mode to execute.
//!
//! This function will allow the CPU01 master system to boot the CPU02 control
//! system via the following modes: Boot to RAM, Boot to Flash, Boot via SPI,
//! SCI, I2C, or parallel I/O. This function blocks and waits until the
//! control system boot ROM is configured and ready to receive CPU01 to CPU02
//! IPC INT0 interrupts. It then blocks and waits until IPC INT0 and
//! IPC FLAG31 are available in the CPU02 boot ROM prior to sending the
//! command to execute the selected bootloader.
//!
//! The \e bootMode parameter accepts one of the following values:
//!  - \b C1C2_BROM_BOOTMODE_BOOT_FROM_PARALLEL
//!  - \b C1C2_BROM_BOOTMODE_BOOT_FROM_SCI
//!  - \b C1C2_BROM_BOOTMODE_BOOT_FROM_SPI
//!  - \b C1C2_BROM_BOOTMODE_BOOT_FROM_I2C
//!  - \b C1C2_BROM_BOOTMODE_BOOT_FROM_CAN
//!  - \b C1C2_BROM_BOOTMODE_BOOT_FROM_RAM
//!  - \b C1C2_BROM_BOOTMODE_BOOT_FROM_FLASH
//!
//! \return 0 (success) if command is sent, or 1 (failure) if boot mode is
//! invalid and command was not sent.
//
//*****************************************************************************
extern uint16_t Device_bootCPU2(uint32_t ulBootMode);
#endif
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
extern void Example_setResultPass(void);
extern void Example_setResultFail(void);
extern void Example_done(void);

//
// End of file
//
